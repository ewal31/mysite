# powershell -executionpolicy bypass -File .\build.ps1
# docker build -t mysitebuildenvironment:1.0 .
# docker run --rm -it -v ${pwd}/docs:/opt/mysite/docs -p 8000:8000 mysitebuildenvironment:1.0 bash # -c "stack exec mysite watch"
# docker run --rm -it -v ${pwd}/docs:/opt/mysite/docs -v ${pwd}/posts:/opt/mysite/posts -v ${pwd}/templates:/opt/mysite/templates -p 8000:8000 mysitebuildenvironment:1.0 bash # -c "stack exec mysite watch"
docker run --rm -it -v ${pwd}:/opt/mysite -p 8000:8000 mysitebuildenvironment:1.0 bash # -c "stack exec mysite watch"
