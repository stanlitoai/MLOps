Using the docker-compose method fails when I try to log in with the password and username defined in the .yaml.
If I try to log in with default and no password, it apparently resets my zenml active stack to default. (see end of log-file below)
After this, running zenml up --docker works again.
And if I shut that down with zenml down I can also restart the container again with zenml up --docker.
Repeating the process broke everything again and I keep getting Error initializing rest store with URL
So, something seems to break when the server is started and stopped/removed again.

After this I ran zenml clean which brought me back to being able to run zenml up --docker successfully, but docker-compose won't budge.