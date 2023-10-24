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

## TODO

- [ ] make use of the above compression tools
- [ ] make site usable on different screen sizes