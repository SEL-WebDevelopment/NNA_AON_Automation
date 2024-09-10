import functions as fn

def main(ARMAP_input_df, AOV_input_df):
    nna_aon_from_armap = fn.find_nna_aon(ARMAP_input_df)
    nna_aon_formatted = fn.format_nna_aon(nna_aon_from_armap)
    aov_updated = fn.replace_old_nna_aon(nna_aon_formatted, AOV_input_df)
    return

if __name__ == "__main__":
     # SDE
    FEATURE_CLASS_OUPUT = "DBO.ARMAP_Map"
    ARMAP_FEATURE_CLASS_INPUT = "DBO.ARMAP_Field_Dates"
    AOV_FEATURE_CLASS_INPUT = 'Arctic_Observing_Sites'
    
    ARMAP_WORKSPACE = r'C:\\ArcGIS Container\\SDE_Connection_Files\\Sel-gis18-script_dev.sde'
    AOV_WORKSPACE = r'C:\\ArcGIS Container\\Maps\\AOV Observing Sites\\Arctic_Observing_Sites\\Arctic_Observing_Sites.gdb'


    ARMAP_input_df  = fn.feature_class_to_dataframe(ARMAP_WORKSPACE, ARMAP_FEATURE_CLASS_INPUT)
    AOV_input_df  = fn.feature_class_to_dataframe(AOV_WORKSPACE, AOV_FEATURE_CLASS_INPUT)
    
    NNA_AON_CSV_OUTPUT = main(ARMAP_input_df, AOV_input_df)
