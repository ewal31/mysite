from haskell:9.2.7-buster

# Update Image
RUN apt update
RUN apt upgrade -y
RUN stack upgrade

# Install NPM and used packages
WORKDIR /opt/node
RUN curl -O https://nodejs.org/dist/v18.16.0/node-v18.16.0-linux-x64.tar.xz && \
    echo '44d93d9b4627fe5ae343012d855491d62c7381b236c347f7666a7ad070f26548 node-v18.16.0-linux-x64.tar.xz' | sha256sum -c && \
	tar -xf node-v18.16.0-linux-x64.tar.xz && \
	rm node-v18.16.0-linux-x64.tar.xz && \
	ln -s "$(pwd)/node-v18.16.0-linux-x64/bin/npm" '/usr/bin/npm' && \
	ln -s "$(pwd)/node-v18.16.0-linux-x64/bin/node" '/usr/bin/node' && \
	npm -g install svgo@3.0.2 && \
	ln -s "$(pwd)/node-v18.16.0-linux-x64/lib/node_modules/svgo/bin/svgo" '/usr/bin/svgo' && \
	npm -g install clean-css-cli@5.5 && \
	ln -s "$(pwd)/node-v18.16.0-linux-x64/lib/node_modules/clean-css-cli/bin/cleancss" '/usr/bin/cleancss' && \
	npm -g install terser@5.18.0 && \
	ln -s "$(pwd)/node-v18.16.0-linux-x64/lib/node_modules/terser/bin/terser" '/usr/bin/terser'
	
# Install dependencies in a separate layer so they can be cached
WORKDIR /opt/mysite
COPY ./mysite.cabal /opt/mysite/mysite.cabal
COPY ./stack.yaml /opt/mysite/stack.yaml
RUN stack build --only-dependencies

# Build Website Generating Code
COPY . /opt/mysite
RUN stack build

# Build Website
CMD ["stack", "exec", "mysite", "build"]