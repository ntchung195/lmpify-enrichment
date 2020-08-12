import pandas as pd

def readExcelFile(file_name,file_sheet,categories_list):
    df = pd.read_excel(file_name, sheet_name = file_sheet)
    for i in range(len(df)):
        Id = df.loc[i,"Id"]
        if( str(Id) in list(categories_list.keys())):
            categories_list[str(Id)].append(df.loc[i,"category"])
    return categories_list
