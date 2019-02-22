# Setup/Contrib

- remove all previous images/containers
- clone this repo
- `cd` to this repo
- `cp .env.dist .env` and update accordingly
- `docker-compose up -d --build`
- exec into flask container and run `./migrateConfigs.sh` to move tessdata into tessdata-best directory
- navigate to localhost:5000
