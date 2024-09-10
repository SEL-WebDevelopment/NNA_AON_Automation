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
    original_rows = unique_df[unique_df.apply(
        tuple, axis=1).isin(df.apply(tuple, axis=1))]

    # Append only unique rows from formatted_df that are not in other_df
    unique_formatted_df = df[~df.apply(tuple, axis=1).isin(
        original_rows.apply(tuple, axis=1))]

    # Append unique rows from formatted_df to other_df
    updated_df = pd.concat([aov_df, unique_formatted_df], ignore_index=True)
    print("updated_df:", len(updated_df))
    print("convert_csv_fields:", len(convert_csv_fields(updated_df)))
    return convert_csv_fields(aov_df)


def convert_csv_fields(df):
    field_types = {
        'Site_Lat': 'Double',
        'Site_Long': 'Double',
        'Proj_Start_Year': 'Double',
        'Proj_End_Year': 'Double',
    }
    # Loop through the dictionary and convert the specified fields
    for field, field_type in field_types.items():
        if field in df.columns:
            if field_type.lower() == 'double':
                df[field] = pd.to_numeric(
                    df[field], errors='coerce').astype(float)
            elif field_type.lower() == 'text':
                df[field] = df[field].astype(str)
            elif field_type.lower() == 'int':
                df[field] = pd.to_numeric(
                    df[field], errors='coerce').astype('Int64')
            elif field_type.lower() == 'date':
                df[field] = pd.to_datetime(df[field], errors='coerce')
            # Add more types as needed

    df['Proj_Start_Year'] = df['Proj_Start_Year'].apply(
        lambda x: int(x) if not pd.isna(x) else pd.NA)
    df['Proj_End_Year'] = df['Proj_End_Year'].apply(
        lambda x: int(x) if not pd.isna(x) else pd.NA)

    # Save the formatted DataFrame to a new CSV file
    return df


def delete_feature_class(workspace, featureClass):
    feature_class_path = os.path.join(workspace, featureClass)
    try:
        arcpy.DeleteFeatures_management(feature_class_path)
        print("Deleted Feature Class")
    except arcpy.ExecuteError as e:
        print(f"Failed to delete features: {e}")


def create_feature_class_from_dataframe(df, output_gdb, feature_class_name):
    # Set environment settings
    arcpy.env.workspace = output_gdb

    # Create a feature class
    feature_class = arcpy.management.CreateFeatureclass(
        output_gdb, feature_class_name, "POINT", spatial_reference=arcpy.SpatialReference(4326))

    # Add fields to the feature class based on the CSV
    text_fields = []
    long_fields = []

    for column in df.columns:
        if column.lower() == 'site_lat' or column.lower() == 'site_long':
            arcpy.management.AddField(feature_class, column, "Double")
            long_fields.append(column)
        elif column.lower() == 'proj_start_year' or column.lower() == 'proj_end_year':
            arcpy.management.AddField(feature_class, column, "LONG")
            long_fields.append(column)
        else:
            max_length = 255
            # Set the max length based on the field values
            max_val_len = df[column].apply(lambda x: len(
                str(x)) if pd.notnull(x) else 0).max()
            # Adjust the max length within limits
            max_length = max(max_length, min(1000, max_val_len))
            arcpy.management.AddField(
                feature_class, column, "TEXT", field_length=max_length)
            text_fields.append(column)

    # Define the insert cursor fields
    cursor_fields = ['SHAPE@XY'] + long_fields + text_fields

    # Insert rows into the feature class
    with arcpy.da.InsertCursor(feature_class, cursor_fields) as cursor:
        for index, row in df.iterrows():
            latitude = row['Site_Lat']
            longitude = row['Site_Long']
            values = [(longitude, latitude)] + [row[col]
                                                for col in long_fields + text_fields]
            try:
                cursor.insertRow(values)
            except Exception as e:
                print(f"Error inserting row {index}: {e}")

    print(
        f"Feature class '{feature_class_name}' created successfully in {output_gdb}")


def update_feature_class_from_dataframe(df, workspace, feature_class):
    # Define the spatial reference (WGS 1984 - EPSG:4326)
    spatial_reference = arcpy.SpatialReference(4326)
    fields = schema.AOV_Schema
    feature_class_path = os.path.join(workspace, feature_class)

    # Iterate through each row in the DataFrame
    with arcpy.da.InsertCursor(feature_class_path, fields + ['SHAPE@']) as cursor:
        for _, row in df.iterrows():
            # Extract Longitude and Latitude values
            longitude = row['Site_Long']
            latitude = row['Site_Lat']

            # Create the PointGeometry object
            point_geometry = arcpy.PointGeometry(
                arcpy.Point(longitude, latitude), spatial_reference)

            # Create a list of values for other fields
            values = [row[field] for field in fields]

            # Append the point geometry to the values list
            values.append(point_geometry)

            # Insert the row into the feature class
            cursor.insertRow(values)
    print(
        f"Feature class '{feature_class}' created successfully in {workspace}")
