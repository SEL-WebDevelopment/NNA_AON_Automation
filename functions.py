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


# Function to handle NaN values
def handle_nan(value):
    if pd.isna(value):
        return None  # or use a suitable default value
    return value


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
    print("convert_csv_fields:", len(convert_csv_fields(updated_df)))
    return convert_csv_fields(updated_df)


def convert_csv_fields(df):
    field_types = {
        'Site_Lat': 'Double',
        'Site_Long': 'Double',
        'Proj_Start_Year': 'Integer',
        'Proj_End_Year': 'Integer',
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


def print_feature_class_schema(feature_class):
    # Get field information
    fields = arcpy.ListFields(feature_class)

    print(f"Fields in {feature_class}:")
    for field in fields:
        print(
            f"\tField: {field.name}, Type: {field.type}, Length: {field.length}")


def update_feature_class_from_dataframe(df, workspace, feature_class):
    # Define the spatial reference (WGS 1984 - EPSG:4326)
    spatial_reference = arcpy.SpatialReference(4326)
    feature_class_path = os.path.join(workspace, feature_class)

    with arcpy.da.InsertCursor(feature_class_path, schema.AOV_Schema_Feature_Class) as cursor:
        for index, row in df.iterrows():
            # Extract Longitude and Latitude values from the DataFrame
            OBJECTID = handle_nan(row['OBJECTID'])
            Proj_Award = handle_nan(row['Proj_Award'])
            Proj_Funding_Country = handle_nan(row['Proj_Funding_Country'])
            Proj_Funding_Agency = handle_nan(row['Proj_Funding_Agency'])
            Proj_Program_Code = handle_nan(row['Proj_Program_Code'])
            Proj_Start_Year = handle_nan(row['Proj_Start_Year'])
            Proj_End_Year = handle_nan(row['Proj_End_Year'])
            Proj_Title = handle_nan(row['Proj_Title'])
            Proj_AON = handle_nan(row['Proj_AON'])
            Proj_Initiative = handle_nan(row['Proj_Initiative'])
            Proj_Discipline = handle_nan(row['Proj_Discipline'])
            Proj_Institution = handle_nan(row['Proj_Institution'])
            Proj_Contact_Name = handle_nan(row['Proj_Contact_Name'])
            Proj_Contact_Email = handle_nan(row['Proj_Contact_Email'])
            Proj_Contact_Phone = handle_nan(row['Proj_Contact_Phone'])
            Proj_Contact_Role = handle_nan(row['Proj_Contact_Role'])
            Proj_Page_Link = handle_nan(row['Proj_Page_Link'])
            Proj_Metadata_Link = handle_nan(row['Proj_Metadata_Link'])
            Site_Name = handle_nan(row['Site_Name'])
            Site_ID_AOV = handle_nan(row['Site_ID_AOV'])
            Site_ID_Alt1 = handle_nan(row['Site_ID_Alt1'])
            Site_ID_Alt2 = handle_nan(row['Site_ID_Alt2'])
            Site_Country = handle_nan(row['Site_Country'])
            Site_Place = handle_nan(row['Site_Place'])
            Site_Lat = handle_nan(row['Site_Lat'])
            Site_Long = handle_nan(row['Site_Long'])
            Site_Accuracy = handle_nan(row['Site_Accuracy'])
            Site_Depth = handle_nan(row['Site_Depth'])
            Site_Elevation = handle_nan(row['Site_Elevation'])
            Site_Start_Date = handle_nan(row['Site_Start_Date'])
            Site_End_Date = handle_nan(row['Site_End_Date'])
            Site_Type = handle_nan(row['Site_Type'])
            Site_GCMD_Science = handle_nan(row['Site_GCMD_Science'])
            Site_GCMD_Platform = handle_nan(row['Site_GCMD_Platform'])
            Site_GCMD_Instrument = handle_nan(row['Site_GCMD_Instrument'])
            Data_Page_Link1 = handle_nan(row['Data_Page_Link1'])
            Data_Page_Link2 = handle_nan(row['Data_Page_Link2'])
            Data_Metadata_Link = handle_nan(row['Data_Metadata_Link'])
            db_Metadata_Sources = handle_nan(row['db_Metadata_Sources'])
            db_Metadata_Wrangler = handle_nan(row['db_Metadata_Wrangler'])
            db_Date_Created = handle_nan(row['db_Date_Created'])
            db_Date_Export = handle_nan(row['db_Date_Export'])
            db_Date_Modified = handle_nan(row['db_Date_Modified'])
            db_Notes = handle_nan(row['db_Notes'])
            Site_Name_IDs = handle_nan(row['Site_Name_IDs'])
            Site_Lat_Long = handle_nan(row['Site_Lat_Long'])
            Site_Location_Country = handle_nan(row['Site_Location_Country'])
            Site_Abstract = handle_nan(row['Site_Abstract'])
            Site_ID_BAID = handle_nan(row['Site_ID_BAID'])
            Site_Start_Year = handle_nan(row['Site_Start_Year'])
            Site_End_Year = handle_nan(row['Site_End_Year'])

            pointGeometry = arcpy.PointGeometry(arcpy.Point(
                Site_Long, Site_Lat), spatial_reference)

            # Insert the row into the feature class
            cursor.insertRow((
                OBJECTID,
                Proj_Award,
                Proj_Funding_Country,
                Proj_Funding_Agency,
                Proj_Program_Code,
                Proj_Start_Year,
                Proj_End_Year,
                Proj_Title,
                Proj_AON,
                Proj_Initiative,
                Proj_Discipline,
                Proj_Institution,
                Proj_Contact_Name,
                Proj_Contact_Email,
                Proj_Contact_Phone,
                Proj_Contact_Role,
                Proj_Page_Link,
                Proj_Metadata_Link,
                Site_Name,
                Site_ID_AOV,
                Site_ID_Alt1,
                Site_ID_Alt2,
                Site_Country,
                Site_Place,
                Site_Lat,
                Site_Long,
                Site_Accuracy,
                Site_Depth,
                Site_Elevation,
                Site_Start_Date,
                Site_End_Date,
                Site_Type,
                Site_GCMD_Science,
                Site_GCMD_Platform,
                Site_GCMD_Instrument,
                Data_Page_Link1,
                Data_Page_Link2,
                Data_Metadata_Link,
                db_Metadata_Sources,
                db_Metadata_Wrangler,
                db_Date_Created,
                db_Date_Export,
                db_Date_Modified,
                db_Notes,
                Site_Name_IDs,
                Site_Lat_Long,
                Site_Location_Country,
                Site_Abstract,
                Site_ID_BAID,
                Site_Start_Year,
                Site_End_Year,
                pointGeometry
            ))

    del cursor

    print(
        f"Feature class '{feature_class}' created successfully in {workspace}")


def replace_feature_class_content(source_fc, dest_fc):
    # Check if the source feature class exists
    if not arcpy.Exists(source_fc):
        raise OSError(f"Source feature class '{source_fc}' does not exist.")

    # Check if the destination path exists
    dest_path = os.path.dirname(dest_fc)
    if not os.path.exists(dest_path):
        raise OSError(f"Destination path '{dest_path}' does not exist.")

    # Check if the destination feature class exists
    if arcpy.Exists(dest_fc):
        print(f"Destination feature class '{dest_fc}' exists. Deleting it...")
        # Delete the destination feature class
        arcpy.Delete_management(dest_fc)

    # Create a new feature class with the same schema as the source feature class
    print(f"Creating new feature class '{dest_fc}'...")
    arcpy.CreateFeatureclass_management(
        out_path=dest_path,
        out_name=os.path.basename(dest_fc),
        template=source_fc
    )

    # Copy features from the source feature class to the new destination feature class
    print(f"Copying features from '{source_fc}' to '{dest_fc}'...")
    arcpy.CopyFeatures_management(source_fc, dest_fc)

    print("Content replacement complete.")
