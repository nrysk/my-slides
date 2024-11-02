#!/usr/bin/env node
const path = require('path')
console.log(process.cwd().split(path.sep).at(-1))