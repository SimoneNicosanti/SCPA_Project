ssh snicosanti@160.80.85.52 "rm -r /home/snicosanti/project"
# scp -r ./project snicosanti@160.80.85.52:/home/snicosanti/
rsync -r --update --exclude 'build/*' ./project snicosanti@160.80.85.52:/home/snicosanti/