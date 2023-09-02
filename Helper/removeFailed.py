import os

def remove_unlisted_csv_files(folder_path, allowed_files):
    removed_count = 0
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv') and file not in allowed_files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                removed_count += 1
    
    return removed_count

if __name__ == "__main__":
    target_folder = "./ambigious"
    allowed_files = ["63c571b3e619a088e65791c1.csv", "631f1b608af38f654d2a3b1f.csv", "6431b44f10493845f0904891.csv", 
"5c257b7dda51990001e2787c.csv", "5ed73d9f9d89251b74a41130.csv", "611b1f8d65d3749d42e746c2.csv", 
"6100652c5329da76a962cdb3.csv", "5d63f9cd26a2ac001758c70d.csv", "5a8cec67aa46dd00016be8c4.csv", 
"5e8644dc6d54ca0a09510c88.csv", "615ddab1e4f013092538b6c5.csv", "62d133496cf7f774b8bf98a0.csv", 
"57f08d467f43070001c3a606.csv", "5c475608cae0ab000188cb6e.csv", "6102053795bd9b55ab979a48.csv", 
"5c8f9fa8dfbf30001697845d.csv", "5ceff90683416100197cde83.csv", "5ff39ec71d6cb49c6c906727.csv", 
"5cc8cab1f67ce800171cc5b2.csv", "600e06e28546f25be49539fa.csv", "6317583b85f3f3f9f23f21fd.csv", 
"63b75c4505aa30e817d4f53f.csv", "60fed4a51b41974258558dbe.csv", "62c63c653a49b220cb390bdb.csv", 
"5dd671942b033b5ec8bc97b4.csv", "5d6349209b22da000117b1bd.csv", "5f77b243c99f31316e4cae58.csv", 
"62c63362e7342ad8be294866.csv", "63d41088860131e49a85cfac.csv", "647c65c5acb91c89819c7970.csv", 
"5c95970cd676900016e1a940.csv", "5ee7201004aef94af287ccc5.csv", "6104b03b594162bfe102b6e8.csv", 
"5db9a46ff8e3f9000f88cb85.csv", "5c745295821a97001763fb1b.csv", "63bc83e58f58b8076e7fad9b.csv", 
"5edabe1c28a45e161cd15325.csv", "5cf6413fca49dd0019155ce0.csv", "5563984afdf99b672b5749b6.csv", 
"583748dafb65a0000128b059.csv", "5efadaebe1d35693595a5f9c.csv@email.prolific.co.csv", "5dcb685070d51c8275d7bd54.csv", 
"60d018cc0f3faeaec2f7b7e5.csv", "6111a8e5c22c39a1c92edb70.csv", "63ebd1cd6715f8cbe3405c82.csv", 
"5f8cbc5c355ea745e6cef2ca.csv", "612962f44f151ddfd0298c52.csv", "63d7e728502b1a8795abd2b2.csv", 
"60fcd2e26a72c518d0b60738.csv", "58adfc6e7cf56d0001f931a2.csv", "63052f20afc39e87110eb83e.csv", 
"63d68e4f60c155f4f1fd9681.csv", "5d7fd2a5987f70001602d95b.csv", "5db4f0b63e33f2000dd54016.csv", 
"58935d5d4d77be0001689f14.csv", "63f77b9ca8439f6e1620c1cd.csv", "63d1b0363f9bd5a6062dfb1c.csv", 
"62bb42e2f21cfdb280cc975f.csv", "61502f4029bebb130459f0ff.csv", "545d3e2bfdf99b7f9e3255b4.csv", 
"5e6af189bdaa1a0b355f63ae.csv", "5dbe5d44aab19f35c5459b57.csv", "5ca749ff98b35f0001f7559d.csv", 
"5b129b48444cef0001cad497.csv", "5bd9dc9abf1a7800011db1f1.csv", "56d0f57421cd29000a9737e4.csv", 
"61675b4c7b3c9e4cc2dc063f.csv", "5d012930cfeb82001817c9c9.csv", "614f89a9441723072040d4de.csv", 
"615efd1df11a7d21f0f12f68.csv", "62d43cee3d60ac98c1dcacc8.csv", "5fbe9241779d54016d6872be.csv", 
"5d4e212b81ca1c001b21672c.csv", "643d780ad443f7f6457d2c3c.csv", "6412285c18716831333d53b3.csv", 
"5f2042752276432e3a7ef104.csv", "5ea7f33ffe8e701e31547364.csv", "64d5219df0415c9a7c609c19.csv", 
"5d7ff85ffec5620017b07c40.csv", "5f5452df72345118f58d187b.csv", "6406300d0af7758269c8dc0b.csv", 
"64135e3d9cd5e3f16290d29c.csv", "62bd1d1283b326cde10cf27a.csv", "5dd4767b32288646716dc98a.csv", 
"62e05a40c77214b0772d5596.csv", "62fb84322433c8d7e890a145.csv", "5d88db42c1d06e001a1fdcab.csv", 
"5f778e4779a4422e3a29a2cf.csv", "5d9be99fdacf6100182bf254.csv", "639695be390aad5b2a2f18f4.csv", 
"6101f4093ed452ae3b6d5ab9.csv", "5fb96b9798cd084465b0bf36.csv", "64528f4d51f889c0c30573af.csv", 
"5dfecf8a814bbab5e1865195.csv", "611622ddd8be1ac51298cb89.csv", "5f4a835e9e84256238129995.csv", 
"5cc99b0973d6e70001c2fc87.csv", "6317ac941faaf331c4573eab.csv", "6112c911a83cc494df38b468.csv", 
"615bb35e52d50aa9a2ae6747.csv", "628270e095732de2fb39cc2a.csv", "60189713dcf864178f74a1ce.csv", 
"63d40e3588ba6ef2a3ce7c37.csv", "63f79194eb27c9dc523185bd.csv", "6006a7b7f57a4801ea3a8323.csv", 
"5dd370b6e4bf2035a6b93643.csv", "62f90c79da5c7deee5339c22.csv", "6496f9fbc2928ee6956ef623.csv", 
"60fdde0f96111e25a7f28205.csv", "5a78b8355292b800012284ca.csv", "60fde7eb8e3e931f2e24dbe0.csv", 
"55b2d3f2fdf99b525bc839aa.csv", "6444b1e677ebe8b4e2c23abd.csv", "5e9c8deb90dd470441c7f98e.csv"]  # Add your correct .csv file names here
    
    removed_count = remove_unlisted_csv_files(target_folder, allowed_files)
    print(f"Removed {removed_count} .csv files.")