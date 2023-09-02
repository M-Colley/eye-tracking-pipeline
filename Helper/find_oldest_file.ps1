
# Get all files from the current directory and subdirectories
$allFiles = Get-ChildItem -Recurse | Where-Object {! $_.PSIsContainer}

# Sort the files by the LastWriteTime property in ascending order
$sortedFiles = $allFiles | Sort-Object LastWriteTime

# Get the first file (the oldest one)
$oldestFile = $sortedFiles[0]

# Display the full path of the oldest file
$oldestFile.FullName
