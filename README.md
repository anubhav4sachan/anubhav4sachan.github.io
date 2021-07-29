# GitHub Portfolio Page

`anubhav4sachan.github.io` is the codebase for my portfolio page and is associated with the domain [anubhavsachan.com](https://anubhavsachan.com).

### Previous Version(s)
`hyde-initial` branch is actively updated or whenever something significant happens.

`master` branch has (very) old website.


## Installing Jekyll and serving the website

```
sudo apt-get install ruby-full build-essential zlib1g-dev
echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc && echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc && echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
cd anubhav4sachan.github.io/
gem install jekyll-paginate jekyll-gist redcarpet
```

Serve the website now by:

`jekyll serve --incremental`

#### Note for WSL (Especially for repo in mounted drives (eg.: `/mnt/c/`)

- For EPERM Error: Shutdown WSL with `wsl --shutdown` in Powershell, then log in to Ubuntu and modify the permissions with `chmod 777 anubhav4sachan.github.io`. [Link 1](https://stackoverflow.com/questions/57243299/jekyll-operation-not-permitted-apply2files/57281081), [Link2](https://stackoverflow.com/questions/46610256/chmod-wsl-bash-doesnt-work)
- Auto-regeneration should work, if it doesn't, use `jekyll serve --force_polling --livereload`.