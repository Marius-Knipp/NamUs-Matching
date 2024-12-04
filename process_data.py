import numpy as np
import pandas as pd
import json
import warnings

warnings.filterwarnings("ignore", message="libuv only supports millisecond timer resolution", category=UserWarning)


with open('./output/MissingPersons.json') as f:
    missingPersons = json.load(f)

missingPersonDescriptions = {}

for i in range(len(missingPersons)):
    id = missingPersons[i].get('idFormatted', 'nan')

    name = missingPersons[i]['subjectIdentification'].get('firstName', 'nan') + ' ' + missingPersons[i]['subjectIdentification'].get('lastName', 'nan')

    # Year when the person went missing
    dateMissing = missingPersons[i]['sighting'].get('date', 'nan')
    
    missingAgeMin = missingPersons[i]['subjectIdentification'].get('computedMissingMinAge', 'nan')
    missingAgeMax = missingPersons[i]['subjectIdentification'].get('computedMissingMaxAge', 'nan')
    
    currentAgeMin = missingPersons[i]['subjectIdentification'].get('currentMinAge', 'nan')
    currentAgeMax = missingPersons[i]['subjectIdentification'].get('currentMaxAge', 'nan')

    heightInches = missingPersons[i]['subjectDescription'].get('heightFrom', 'nan')

    weightLbs = missingPersons[i]['subjectDescription'].get('weightFrom', 'nan')

    ethnicity = ', '.join(feature.get('name', 'nan') for feature in missingPersons[i]['subjectDescription']['ethnicities'])
    primaryEthnicity = missingPersons[i]['subjectDescription']['primaryEthnicity'].get('name', 'nan')

    sex = missingPersons[i]['subjectDescription']['sex'].get('name', 'nan')

    clothing = '\n'.join([feature.get('description', 'nan') for feature in missingPersons[i]['clothingAndAccessoriesArticles']])

    location_city = missingPersons[i]['sighting']['address'].get('city', 'nan')
    location_state = missingPersons[i]['sighting']['address']['state'].get('name', 'nan')
    city_state = f"{location_city}, {location_state}"
    
    if 'publicGeolocation' in missingPersons[i]['sighting']:
        location = (float(missingPersons[i]['sighting']['publicGeolocation']['coordinates']['lat']),
                    float(missingPersons[i]['sighting']['publicGeolocation']['coordinates']['lon']))
    else:
        location = 'nan'

    if 'physicalDescription' in missingPersons[i]:
        hairColor = missingPersons[i]['physicalDescription']['hairColor'].get('name', 'nan')

        leftEyeColor = missingPersons[i]['physicalDescription']['leftEyeColor'].get('name', 'nan')

        rightEyeColor = missingPersons[i]['physicalDescription']['rightEyeColor'].get('name', 'nan')

        physicalFeatures = '\n'.join([f"{feature['physicalFeature']['name']}: {feature.get('description', 'nan')}" 
                                    for feature in missingPersons[i]['physicalFeatureDescriptions']])
    
    person = {
        'Last Contact': dateMissing,
        'City State': city_state,
        'Location': location,
        'Sex': sex,
        'Missing Age From': missingAgeMin,
        'Missing Age To': missingAgeMax,
        'Current Age From': currentAgeMin,
        'Current Age To': currentAgeMax,
        'Ethnicity': primaryEthnicity,
        'Height': heightInches,
        'Weight': weightLbs,
        'Hair Color': hairColor,
        'Left Eye Color': leftEyeColor,
        'Right Eye Color': rightEyeColor,
        'Clothing': clothing,
        'Physical Features': physicalFeatures,
    }
    
    missingPersonDescriptions[id] = person
    
# Convert the nested dictionary to a pandas DataFrame and save it to a csv file
missing_persons_df = pd.DataFrame.from_dict(missingPersonDescriptions, orient='index')
missing_persons_df.index.name = 'ID'



with open('./output/UnidentifiedPersons.json') as f:
    unidentifiedPersons = json.load(f)

unidentifiedPersonDescriptions = {}

for i in range(len(unidentifiedPersons)):
    
    id = unidentifiedPersons[i].get('idFormatted', 'nan')
    
    dateFound = unidentifiedPersons[i]['circumstances'].get('dateFound', 'nan')
    
    
    estimatedAgeFrom = unidentifiedPersons[i]['subjectDescription'].get('estimatedAgeFrom', 'nan')
    estimatedAgeTo = unidentifiedPersons[i]['subjectDescription'].get('estimatedAgeTo', 'nan')
    if 'estimatedAgeGroup' in unidentifiedPersons[i]['subjectDescription']:
        ageGroup = unidentifiedPersons[i]['subjectDescription']['estimatedAgeGroup'].get('name', 'nan')
    else:
        ageGroup = 'nan'

        
    estimatedYearOfDeathFrom = unidentifiedPersons[i]['subjectDescription'].get('estimatedYearOfDeathFrom', 'nan')
    estimatedYearofDeathTo = unidentifiedPersons[i]['subjectDescription'].get('estimatedYearOfDeathTo', 'nan')

    heightInches = unidentifiedPersons[i]['subjectDescription'].get('heightFrom', 'nan')

    weightLbs = unidentifiedPersons[i]['subjectDescription'].get('weightFrom', 'nan')
    
    primaryEthnicity = unidentifiedPersons[i]['subjectDescription']['primaryEthnicity'].get('name', 'nan')

    sex = unidentifiedPersons[i]['subjectDescription']['sex'].get('name', 'nan') if 'sex' in unidentifiedPersons[i]['subjectDescription'] else 'nan'

    clothing = '\n'.join([feature.get('description', 'nan') for feature in unidentifiedPersons[i]['clothingAndAccessoriesArticles']
                          if feature.get('description', 'nan') != 'none'])
    
    if 'publicGeolocation' in unidentifiedPersons[i]['circumstances']:
        location = (float(unidentifiedPersons[i]['circumstances']['publicGeolocation']['coordinates']['lat']),
                    float(unidentifiedPersons[i]['circumstances']['publicGeolocation']['coordinates']['lon']))
    else:
        location = 'nan'

    
    if 'physicalDescription' in unidentifiedPersons[i]:
    
        hairColor = (unidentifiedPersons[i]['physicalDescription']['hairColor']['name'] 
                    if 'hairColor' in unidentifiedPersons[i]['physicalDescription'] 
                    else 'nan')
        
        if 'leftEyeColor' in unidentifiedPersons[i]['physicalDescription']:
            leftEyeColor = unidentifiedPersons[i]['physicalDescription']['leftEyeColor'].get('name', 'nan')
        else:
            leftEyeColor = 'nan'

        if 'rightEyeColor' in unidentifiedPersons[i]['physicalDescription']:
            rightEyeColor = unidentifiedPersons[i]['physicalDescription']['rightEyeColor'].get('name', 'nan')
        else:
            rightEyeColor = 'nan'

    physicalFeatures = '\n'.join([f"{feature['physicalFeature']['name']}: {feature.get('description', 'nan')}" 
                                  for feature in unidentifiedPersons[i]['physicalFeatureDescriptions']])
    
    person = {
        'Date Found': dateFound,
        'Location': location,
        'Sex': sex,
        'Estimated Age From': estimatedAgeFrom,
        'Estimated Age To': estimatedAgeTo,
        'Age Group': ageGroup,
        'Estimated Year of Death From': estimatedYearOfDeathFrom,
        'Estimated Year of Death To': estimatedYearofDeathTo,
        'Ethnicity': primaryEthnicity,
        'Height': heightInches,
        'Weight': weightLbs,
        'Hair Color': hairColor,
        'Left Eye Color': leftEyeColor,
        'Right Eye Color': rightEyeColor,
        'Clothing': clothing,
        'Physical Features': physicalFeatures,
    }
    
    unidentifiedPersonDescriptions[id] = person

# Convert the nested dictionary to a pandas DataFrame
unidentified_persons_df = pd.DataFrame.from_dict(unidentifiedPersonDescriptions, orient='index')
unidentified_persons_df.index.name = 'ID'



# Define a function to replace non-standard missing values with NaN
def replace_non_standard_missing_values(df, columns, missing_indicators):
    for column in columns:
        for indicator in missing_indicators:
            df[column] = df[column].replace(indicator, np.nan)
    return df

# List of non-standard missing value indicators
missing_value_indicators = ["Unsure", "Uncertain", "Cannot Determine", "Other", 
                            "Unknown", "Missing Eye", "Multiple", "None", "nan", "N/A", "-", " - ", ""]

missing_persons_df = replace_non_standard_missing_values(missing_persons_df, missing_persons_df.columns[1:], missing_value_indicators)
unidentified_persons_df = replace_non_standard_missing_values(unidentified_persons_df, unidentified_persons_df.columns[1:], missing_value_indicators)

missing_persons_df.to_csv('./output/MissingPersons_filtered.csv')
unidentified_persons_df.to_csv('./output/UnidentifiedPersons_filtered.csv')