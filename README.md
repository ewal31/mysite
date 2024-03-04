# mysite

This is the source code for [my website](https://ewal31.github.io/mysite/).

## Build

The most straightforward way to build the project is to use the provided [Dockerfile](./Dockerfile).

```bash
docker build -v ${pwd}/docs:/opt/mysite/docs .
```

The resulting website that is then provided by Github Pages is located within the `docs` folder.

## Tools for reducing compiled size

- https://github.com/svg/svgo
- https://github.com/clean-css/clean-css-cli
- https://www.npmjs.com/package/terser

## Requirements

* zlib1g-dev

## TODO

**High Priority**

- [ ] rework the github action to build the project itself, so that the doc folder doesn't
      have to be checked in and the repository size can be reduced. (especially adding photos)
- [ ] nix in place of docker for building


**Low Priority**

- [ ] make use of the above compression tools
    - [x] javascript
    - [ ] css
    - [ ] svg
- [x] make site usable on different screen sizes
- [ ] clean up presentation of references
- [ ] purescript instead of javascript
- [ ] perhaps change sidenotes, to fix to html markers in the text
- [ ] add contents for posts
- [ ] support dark mode
- [ ] support different text sizes
- [ ] remove extra newline added in sidenotes
