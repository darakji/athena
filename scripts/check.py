from ase.io import read
import numpy as np
li_path = "/home/mehuldarak/athena/bulk_references/Li_100_slab__LLZO_001_Zr_code93_sto/bulk_li__Li_100_slab__LLZO_001_Zr_code93_sto__c1__ip1x5.cif"
llzo_path = "/home/mehuldarak/athena/archive/bulk_references/Li_100_slab__LLZO_001_Zr_code93_sto/bulk_llzo_vacancy__Li_100_slab__LLZO_001_Zr_code93_sto__nvac14.cif"

li   = read(li_path)
llzo = read(llzo_path)

print("Li   a b c:", np.linalg.norm(li.cell[0]),   
                     np.linalg.norm(li.cell[1]),   
                     np.linalg.norm(li.cell[2]))
print("LLZO a b c:", np.linalg.norm(llzo.cell[0]), 
                     np.linalg.norm(llzo.cell[1]), 
                     np.linalg.norm(llzo.cell[2]))
print("Li   N atoms:", len(li))
print("LLZO N atoms:", len(llzo))