# Setup/Contrib

- remove all previous images/containers
- clone this repo
- `cd` to this repo
- `docker-compose up -d`
- To exec into the container:
    - Windows:
        - `winpty docker exec -it capstone_backend bash`
    - Linux/Unix/Mac
        - `docker exec -it capstone_backend bash`
