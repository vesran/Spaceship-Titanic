# Spaceship Titanic

My take on the <a href='https://www.kaggle.com/competitions/spaceship-titanic'>Spaceship Titanic</a> challenge from Kaggle. 
Here is exposed the solution to generate a submission file with a score of 0.8099 on the leaderboard (top 100/+2300).

### How to use it ?

```shell
# Start container first
docker-compose up -d
docker attach spaceship_lab_1

# Create a submission
make submission

# Start Jupyter
make lab
```

Submission files are stored in `data/output`.