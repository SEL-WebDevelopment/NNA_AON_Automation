import arcpy
import os
import pandas as pd
import schema as schema


def feature_class_to_dataframe(workspace, feature_class_input):
    feature_class_path = os.path.join(workspace, feature_class_input)
    # Use a try-except block to handle errors
    try:
        # Check if arcpy is available
        if arcpy is None:
            raise ImportError(
                "arcpy module is not available. Please make sure you have ArcGIS installed.")

        # Check if the feature class exists
        if not arcpy.Exists(feature_class_path):
            raise ValueError(
                "Feature class does not exist: {}".format(feature_class_path))

        # Initialize an empty list to store the data
        data = []
        # Get a list of field names
        field_names = [field.name for field in arcpy.ListFields(
            feature_class_path) if field.name != 'Shape']

        # Use a SearchCursor to iterate over rows in the feature class
        with arcpy.da.SearchCursor(feature_class_path, field_names) as cursor:
            for row in cursor:
                data.append(row)

        # Convert the data into a Pandas DataFrame
        df = pd.DataFrame(data, columns=field_names)

        return df

    except Exception as e:
        print("An error occurred:", str(e))
        return None


def find_nna_aon(df: pd.DataFrame) -> pd.DataFrame:
    # List of words to match
    words_to_match = ['NNA', 'AON']
    # Columns to search in
    columns_to_search = ['Project_Title', 'Program']

    # Create a pattern to search for words
    pattern = '|'.join(words_to_match)
    # Initialize a boolean mask to False for all rows
    mask = pd.Series([False] * len(df))

    # Update the mask if any word is found in any of the specified columns
    for col in columns_to_search:
        mask |= df[col].str.contains(pattern, case=False, na=False)

    print(f"Total NNA/AON records found: {len(df[mask])}")

    # Return the filtered DataFrame
    return df[mask]


def format_nna_aon(df: pd.DataFrame) -> pd.DataFrame:
    # Iterate through each required field and assign the hardcoded value if it exists in default_values
    for field in schema.AOV_Schema:
        df[field] = schema.aov_hardcode_values.get(field, df.get(field, ''))

    # Move content to each column
    df['Proj_Award'] = df['Award_Num']
    df['Proj_Contact_Name'] = df['PI']
    df['Proj_Funding_Agency'] = df['Funding_Agency']
    df['Proj_Program_Code'] = df['Program']
    df['Proj_Title'] = df['Project_Title']
    df['Proj_Discipline'] = df['Discipline']
    df['Proj_Start_Year'] = df['Start_Year']
    df['Proj_End_Year'] = df['End_Year']
    df['Proj_Contact_Email'] = df['PI_Email']
    df['Proj_Contact_Phone'] = df['PI_Phone']
    df['Site_Lat'] = df['Latitude']
    df['Site_Long'] = df['Longitude']
    df['Site_Abstract'] = df['Abstract']
    df['Site_Place'] = df['Place']
    df['Proj_Page_Link'] = 'https://api.battellearcticgateway.org/v1/reports/grant?proposal_number='+df['Award_Num']

    # Keep only the required fields in the DataFrame
    df = df[schema.AOV_Schema]
    return df


def replace_old_nna_aon(df, aov_df):
    # Ensure the DataFrames have the same columns
    if set(df.columns) != set(aov_df.columns):
        raise ValueError("Both DataFrames must have the same columns")

    # Concatenate the two DataFrames
    combined_df = pd.concat([aov_df, df], ignore_index=True)

    # Drop duplicates based on all columns
    unique_df = combined_df.drop_duplicates(keep='first')

    # Identify the rows in unique_df that were originally from formatted_df
    # Using `isin` requires tuple conversion for row comparisons
    original_rows = unique_df[unique_df.apply(tuple, axis=1).isin(df.apply(tuple, axis=1))]

    # Append only unique rows from formatted_df that are not in other_df
    unique_formatted_df = df[~df.apply(tuple, axis=1).isin(original_rows.apply(tuple, axis=1))]

    # Append unique rows from formatted_df to other_df
    updated_df = pd.concat([aov_df, unique_formatted_df], ignore_index=True)
    print(len(updated_df))

    return updated_df
    

