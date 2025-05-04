allow training based off of folder of files 
GUID for each run
save run stats for GUID in txt file
compare output at each visualization interval, if they're the same then change the entropy
when training on a group of files, include the name of each image in the metadata for training. We want to later be able to ask the model to generate an image that most closely resembles the name we provide

python src/train.py --async-visualization --optimize-memory --visualization-interval 500 --epochs 1000