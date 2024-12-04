import os
import pandas as pd
import ast
import numpy as np
from geopy.distance import geodesic
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm.auto import tqdm

pd.options.mode.chained_assignment = None


# Constants
HEIGHT_TOLERANCE = 3

# Lambda parameters
LAMBDA_TIME = 0.15
LAMBDA_LOCATION = 0.0025
LAMBDA_WEIGHT = 0.025

# There are 9 information types
n_overlapping_information = 7
OVERLAPPING_INFORMATION_THRESHOLD = n_overlapping_information / 9

# Hugging Face Token
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Load the model and tokenizer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
TOKENIZER.pad_token_id=TOKENIZER.eos_token_id

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
MODEL = AutoModelForCausalLM.from_pretrained(model_id, 
                                             token=HF_TOKEN,
                                             device_map='auto',
                                             quantization_config=quantization_config)



class PersonMatcher:
    def __init__(self, missing_persons_df, unidentified_persons_df, top_n, batch_size=32, gpu=False):
        self.missing_persons_df = missing_persons_df
        self.unidentified_persons_df = unidentified_persons_df
        self.top_n = top_n  # How many matches to return for each MP
        self.batch_size = batch_size
        self.gpu = gpu 
    
    def exponential_decay_similarity(self, diff, lambda_param):
        return np.exp(-lambda_param * diff)
    
    def calculate_distance(self, from_location, to_location):
        if pd.isna(from_location) or pd.isna(to_location):
            return np.nan
        from_location = ast.literal_eval(from_location)
        to_location = ast.literal_eval(to_location)
        return geodesic(from_location, to_location).km

    def transform_scale(self, old_rating):
        min_old, max_old = 1, 5
        min_new, max_new = 0, 1
    
        # Apply the transformation formula
        new_rating = (old_rating - min_old) / (max_old - min_old) * (max_new - min_new) + min_new
    
        return new_rating
    
    def dynamic_average(self, scores):
        return scores.mean(skipna=True)

    def check_nan_overlap(self, mp_row, col_pairs):
        # Initialize a DataFrame to hold comparison results
        comparison_results = pd.DataFrame(index=self.unidentified_persons_df.index)
        
        # Loop through each column to compare
        for col in col_pairs:
            # Check if the value in mp_row for this column is NaN
            mp_val_is_nan = pd.isna(mp_row[col])
            
            # Create a boolean Series that is True where either the mp_row value is NaN
            # or the value in the unidentified_persons_df for this column is NaN
            comparison_results[col] = self.unidentified_persons_df[col].isna() | mp_val_is_nan
            
        # After comparing all columns, calculate the share of columns with at least one NaN
        overlapping_information_share = 1 - comparison_results.mean(axis=1)
    
        return overlapping_information_share
        
    def apply_must_match_criteria(self, mp_row, up_df):
        
        '''
        This function filters the unidentified persons DataFrame based on must-match criteria.
        Note: If at least one of the must-match criteria of the missing person is NaN, there will be no match.
        '''
               
        # up_df = self.unidentified_persons_df
        
        # Calculate the age of the missing person at the time the body was found
        years_elapsed = (up_df['Date Found'] - mp_row['Last Contact']).dt.days / 365.25
        mp_age_at_body_found = mp_row['Missing Age From'] + years_elapsed
        
        # Apply filtering based on must-match criteria
        filtered_unidentified_persons_df = up_df[
            (
                (up_df['Date Found'] >= mp_row['Last Contact']) | (up_df['Date Found'].isna())
            ) &
            (
                (mp_row['Last Contact'].year <= up_df['Estimated Year of Death To']) | (up_df['Estimated Year of Death To'].isna())
            ) &
            (
                (up_df['Estimated Age From'].le(mp_age_at_body_found) & up_df['Estimated Age To'].ge(mp_age_at_body_found)) |
                (up_df['Estimated Age From'].isna()) | (up_df['Estimated Age To'].isna())
            ) &
            (
                (up_df['Sex'] == mp_row['Sex']) | (up_df['Sex'].isna())
            ) &
            (
                (up_df['Height'].between(mp_row['Height'] - HEIGHT_TOLERANCE, mp_row['Height'] + HEIGHT_TOLERANCE)) |
                (up_df['Height'].isna())
            ) &
            (
                (up_df['Left Eye Color'] == mp_row['Left Eye Color']) | (up_df['Left Eye Color'].isna()) |
                (up_df['Right Eye Color'] == mp_row['Right Eye Color']) | (up_df['Right Eye Color'].isna())
            ) &
            (
                (up_df['Ethnicity'] == mp_row['Ethnicity']) | (up_df['Ethnicity'].isna())
            )
        ]
        
        return filtered_unidentified_persons_df

    
    def apply_similarity_scores(self, filtered_df, mp_row):
        # Weight similarity
        filtered_df['Weight Similarity'] = filtered_df['Weight'].apply(
            lambda x: self.exponential_decay_similarity(abs(x - mp_row['Weight']) if pd.notna(x) else np.nan, LAMBDA_WEIGHT)
        )
        
        # Weight similarity score > 0.05 (i.e. less than ~100lbs difference)
        filtered_df = filtered_df[filtered_df['Weight Similarity'] > 0.05]
        
        # Time similarity
        filtered_df['Time Similarity'] = filtered_df['Date Found'].apply(
            lambda x: self.exponential_decay_similarity(((x - mp_row['Last Contact']).days / 365.25) if pd.notna(x) else np.nan, LAMBDA_TIME)
        )

        # Location similarity
        filtered_df['Location Similarity'] = filtered_df['Location'].apply(
            lambda x: self.exponential_decay_similarity(self.calculate_distance(x, mp_row['Location']), LAMBDA_LOCATION)
        )   
        
        if self.gpu:
            filtered_df['Physical Features Similarity'] = self.ask_model_feature_similarity(
                mp_row['Physical Features'], filtered_df['Physical Features'], 'physical feature')
            
            filtered_df['Clothing Similarity'] = self.ask_model_feature_similarity(
                mp_row['Clothing'], filtered_df['Clothing'], 'clothing')
        
        return filtered_df

    def ask_model_feature_similarity(self, mp_feature, up_df, feature):
        prompts = []

        for up_feature in up_df:
            if pd.isna(mp_feature) or pd.isna(up_feature):
                prompts.append(None)
                continue

            # Blocker - Check if at least one token is the same
            mp_feature_tokens = set(TOKENIZER(mp_feature, add_special_tokens=False).input_ids)
            up_feature_tokens = set(TOKENIZER(up_feature, add_special_tokens=False).input_ids)
            common_tokens = mp_feature_tokens.intersection(up_feature_tokens)
            if not common_tokens:
                prompts.append(None)
                continue
                
            mp_feature = mp_feature.lower()
            up_feature = up_feature.lower()

            prompt = f'''These are the {feature} descriptions of a missing persons and an unidentified persons. \
Rate the similarity of the {feature} descriptions on a scale between 1 to 5 where 1 is not similar at all and 5 is very similar. \
Answer only with the score, do not include any other information.

Missing Person: "{mp_feature}"

Unidentified Person: "{up_feature}"

Similarity Score:'''

            prompts.append(prompt)

        # Remove prompts with missing values
        valid_prompts = [p for p in prompts if p is not None]
        
        # Batch processing with the model
        responses = []
        num_batches = len(valid_prompts) // self.batch_size
        
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size
            batch_prompts = valid_prompts[start_idx:end_idx]
            batch_inputs = TOKENIZER(batch_prompts, return_tensors='pt', padding=True).to(DEVICE)
            with torch.no_grad():
                batch_outputs = MODEL.generate(**batch_inputs, max_new_tokens=2, temperature=0.01, do_sample=True, pad_token_id=TOKENIZER.eos_token_id)
            batch_response = TOKENIZER.batch_decode(batch_outputs, skip_special_tokens=True)
            for response in batch_response:
                responses.append(response)

        remaining_prompts = valid_prompts[num_batches * self.batch_size:]
        if remaining_prompts:
            batch_inputs = TOKENIZER(remaining_prompts, return_tensors='pt', padding=True).to(DEVICE)
            with torch.no_grad():
                batch_outputs = MODEL.generate(**batch_inputs, max_new_tokens=2, temperature=0.01, do_sample=True, pad_token_id=TOKENIZER.eos_token_id)
            batch_response = TOKENIZER.batch_decode(batch_outputs, skip_special_tokens=True)
            for response in batch_response:
                responses.append(response)
        
        response_idx = 0
        similarity_scores = []
        for prompt in prompts:
            if prompt is None:
                answer = np.nan
            else:
                response = responses[response_idx]
                answer = response[-1]
                try:
                    answer = int(answer)
                    answer = self.transform_scale(answer)  # Transform scale from 1-5 to 0-1
                except ValueError:
                    answer = np.nan
                response_idx += 1
            
            similarity_scores.append(answer)

        return similarity_scores
    
    def find_matches(self):
        ''' Main function to find matches between missing persons and unidentified persons '''
        
        matches_df = pd.DataFrame()
        
        # Iterate over each missing person row with a progress bar
        for i, mp_row in tqdm(self.missing_persons_df.iterrows(), total=self.missing_persons_df.shape[0]):
            # Check overlapping information and keep only rows with a certain threshold
            col_pairs = [col for col in mp_row.index if col in self.unidentified_persons_df.columns if col not in ['ID', 'Hair Color']]
            overlapping_information_share = self.check_nan_overlap(mp_row, col_pairs)
            filtered_unidentified_persons = self.unidentified_persons_df[overlapping_information_share >= OVERLAPPING_INFORMATION_THRESHOLD]
            
            # Apply must-match criteria to filter potential matches
            filtered_unidentified_persons = self.apply_must_match_criteria(mp_row, filtered_unidentified_persons)

            # Check if there are any potential matches left after filtering
            if not filtered_unidentified_persons.empty:

                # Apply similarity scores for each potential match
                filtered_unidentified_persons = self.apply_similarity_scores(filtered_unidentified_persons, mp_row)

                # List of columns that represent similarity scores
                similarity_columns = [column for column in filtered_unidentified_persons.columns if 'Similarity' in column]

                # Calculate a dynamic average of similarity scores for each potential match
                filtered_unidentified_persons['Similarity Score'] = filtered_unidentified_persons[similarity_columns].apply(
                    self.dynamic_average, axis=1
                )
                
                # Get the top n matches based on the highest similarity scores
                top_matches = filtered_unidentified_persons.nlargest(self.top_n, 'Similarity Score')
                
                # Number of how many fulfilled must-match criteria
                len_matches = len(filtered_unidentified_persons)

                # Iterate over each top match and append to matches_df
                for rank, (index, top_match) in enumerate(top_matches.iterrows(), start=1):
                    current_match_df = pd.DataFrame([{
                        'MP ID': mp_row['ID'],
                        'UP ID': top_match['ID'],
                        'Similarity Score': top_match['Similarity Score'],
                        **{col: top_match[col] for col in similarity_columns}  # Add all similarity columns
                    }])
                    
                    matches_df = pd.concat([matches_df, current_match_df], ignore_index=True)

        # Return the compiled DataFrame of all matches
        return matches_df
    
    
    
missing_persons_df = pd.read_csv('output/MissingPersons_filtered.csv', parse_dates=['Last Contact'])
unidentified_persons_df = pd.read_csv('output/UnidentifiedPersons_filtered.csv', parse_dates=['Date Found'])

matcher = PersonMatcher(missing_persons_df, unidentified_persons_df, top_n=1, batch_size=24, gpu=True)
matches_df = matcher.find_matches().round(2)
matches_df = matches_df.sort_values(by="Similarity Score", ascending=False)

matches_df.to_excel('output/matches.xlsx', index=False)