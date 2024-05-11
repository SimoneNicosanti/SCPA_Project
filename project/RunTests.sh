# Sync Project
ssh snicosanti@160.80.85.52 "rm -r /home/snicosanti/project"
rsync -r --update --exclude 'build/*' ./project snicosanti@160.80.85.52:/home/snicosanti/

# Build Project
ssh snicosanti@160.80.85.52 "cd ./project/out 