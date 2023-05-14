from haskell:9.2.7-buster

WORKDIR /opt/mysite

RUN stack upgrade

# Install dependencies in a separate layer so they can be cached
COPY ./mysite.cabal /opt/mysite/mysite.cabal
COPY ./stack.yaml /opt/mysite/stack.yaml

RUN stack build --only-dependencies

COPY . /opt/mysite

CMD ["stack" "build"]
