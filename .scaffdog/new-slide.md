---
name: 'new-slide'
root: '.'
output: '.'
questions:
  title: 'Enter the title of the new slide'
  conference: 'Enter the conference name'
---

# `slides/{{ inputs.conference != '' ? inputs.conference + '-' : '' }}{{ date 'YYYY-MM-DD' }}/slides.md`

```markdown
---
theme: default
title: {{ inputs.title }}
transition: slide-up
layout: center
---

# {{ inputs.title }}
```

# `slides/{{ inputs.conference != '' ? inputs.conference + '-' : '' }}{{ date 'YYYY-MM-DD' }}/package.json`

```json
{
  "type": "module",
  "private": true,
  "scripts": {
    "build": "slidev build --base $npm_package_name --out ../../dist",
    "dev": "slidev --open",
    "export": "slidev export"
  }
}
```
