# Fixture guide (local-only markers)

A fixture for devkit-site.test.ts: a small DEVKIT.md (every path, folder and name in it is made up) with notes about one
computer in local-only blocks.<!-- local-only:start --> On this machine the repository lives in D:\work\atlas-demo.<!-- local-only:end -->

## Contents

1. [Set up](#1-set-up)
2. [Next](#2-next)

---

## 1. Set up

You need:

- **Node 24** and npm.
- Git.<!-- local-only:start --> On this machine, every git command needs `git -c "safe.directory=D:/work/atlas-demo" ...` (the drive is
  exFAT; see the [machine notes](#notes) below).<!-- local-only:end -->

Links: the [hidden notes](#notes), the [public notes](#notes-1) and [the next section](#2-next).

<!-- local-only:start -->
### Notes

- **Writing files on `D:\`.** Write into the scratchpad first (C:/Users/someone/AppData/Local/Temp), then copy.
- **Python.** The local config points at a made-up virtual environment:

  ```json
  { "python": "D:/work/venvs/demo/Scripts/python.exe" }
  ```
<!-- local-only:end -->

### Notes

Public notes. A marker inside a code block is text, not a marker:

```
<!-- local-only:start -->
```

---

## 2. Next

Text of the next section.
