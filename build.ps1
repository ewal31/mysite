# powershell -executionpolicy bypass -File .\build.ps1
# docker build -t mysitebuildenvironment:1.0 .
docker run --rm -it -v ${pwd}/docs:/opt/mysite/docs mysitebuildenvironment:1.0 bash -c "stack build && stack exec mysite build"
