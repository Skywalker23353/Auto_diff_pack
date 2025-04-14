def write_to_file(write_path, filename, data):
    n = len(data)      
    with open(write_path + "/" + filename + ".txt", 'w') as f:
        f.write("%d\n" % n)
    with open(write_path + "/" + filename + ".txt", 'a') as f:
        for item in data:
            f.write("%e\n" % item)