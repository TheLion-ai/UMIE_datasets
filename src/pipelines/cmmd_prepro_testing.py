import os 

main_path = "C:/Users/LENOVO/Desktop/umie/cmmd_data"
ex_img_path = "C:/Users/LENOVO/Desktop/umie/cmmd_data/manifest-1616439774456/CMMD/D1-0001/07-18-2010-NA-NA-79377/1.000000-NA-70244/1-2.dcm" 

img_id = os.path.basename(ex_img_path) 
img_id = os.path.splitext(img_id)[0].split('-')[1]
print(img_id) 

img_col = os.path.basename(os.path.dirname(ex_img_path)).split('-')[2]
print(img_col)

img_id_final = img_col + '_' + img_id 
print(img_id_final)

two_lvls_higher = os.path.basename(os.path.dirname(os.path.dirname(ex_img_path))).split('-')[5] 
print(two_lvls_higher) 

