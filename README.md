Build the container image:

    docker build -t app .

Start the container:

    docker run --name my_container -p 8050:8050 app
