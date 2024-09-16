# PATTIE_User_Modeling
This private repository will be for the development of user login, modeling, and information management function integration with PATTIE.

```bash
cd DIRECTORY  # your choice of directory
git clone https://github.com/YOUR_USER_NAME/PATTIE_User_Modeling.git #You will need to be authorized to access this repository
```

Install the required software packages from the configuration file "requirements.txt"
```bash
pip3 install -r requirements.txt
```

Then, run a code **locally** as follows.

```bash
cd PATH/TO/DIRECTORY
./run_pattie
```

Then, open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser on your **local** computer.

Push **local** changes to the **online** repository.
```bash
git status # will show you what has been changed and needs to be staged
git add . # will stage your code changes
git commit -m 'write your note on these changes' # this will let the project manager (Michael) what changes you've made.
git push
```

Update the **local** repository.
```bash
git pull # this will bring all of the teams recent code changes to your machine for further testing
```
# Updates 09/21/2020 (About operations regarding staging branches)

To build a minimalist "deployment pipeline," two remote branches, origin/alpha and origin/beta have been created in addition to origin/master. Alpha branch is for development and beta branch is for testing before production, while master branch is production branch.

Actions needed to support the new system:

**First time setup: one-shot actions**

1. Go to your local repo.

```
cd PATH/TO/DIRECTORY
```

2. Update your local repo info to reflect the newly added remote branches:

```
git remote update origin --prune
```
3. Verify the remote information is added

```
git branch -a # should list all the local branches and remote branches
```
4. Setup your local branch to track the alpha remote branch:

```
git branch <local_branch> -u <remote>/<remote_branch> # -u is shorthand for --set-upstream
```
in our case, if your local branch is 'master':

```
git branch master -u origin/alpha
```

5. Verify the new upstream is updated:

```
git status # should say that your local branch is tracking origin/alpha
```

**Every-time actions (upon pushing changes)**


1. Suppose all the remote branches are synced up right now, you have some local changes you want to push to alpha remote branch:

```
git add <files> # stage the local files you want to commit

git commit -m "<commit_message>" # make a commit in your local repo

git push origin master:alpha # push code changes in your local branch (master) to your remote branch (origin/alpha)
```

2. Verify

```
git status # should say that local branch is up to date with origin/alpha
```

Now the alpha remote branch has the latest changes while beta and master (production) are still in there old states. It's very likely you haven't decided to push the changes to beta and master yet, because you still need to do some testings on alpha release. 

To make matters worse, after a while, there are some local commits and code changes in your local repo (testing shouldn't block development), which haven't been pushed to any remote branches yet. 

At one point sooner or later, you'll want to sync your alpha branch with two other branches (alpha has commits newer than other remote branches but older than your local commits,) and here's how you'll do it

1. Temporarily move you local changes away (they will not disappear):

```
git add <files> # stage the local files you want to commit 

git stash # stash command temporarily "hide" all your local changes to keep you repo clean when doing branching operation
```
2. Fetch information from alpha remote branch:

```
git fetch origin/alpha # fetch the alpha commits into a middle ground, doing nothing to your local repo
```
3. Create a new local branch, store all the commits you just fetched to this temporary local branch, and go to this branch:

```
git checkout -b tmp origin/alpha # "checkout" means go to the local branch you specify; "-b" means create the new branch you specify if it does not exist yet; "tmp" is your local branch; "origin/alpha" is your remote branch
```
4. Now that you are in the tmp local branch, which has the exact same commit history as alpha, you can push to other branches, which essentially syncs up alpha with them:

```
git push origin tmp:<remote_branch> # origin is your remote (collection of your remote branches); "tmp" is your local branch

e.g.
git push origin tmp:beta
git push origin tmp:master
```
5. Checkout master branch and delete tmp branch in your local repo:

```
git checkout master # leave tmp local branch, go to master local branch

git branch -d tmp # delete tmp local branch
```
6. Put back the local changes you've "hidden" before and continue to work

```
git stash pop
```

At this point, your alpha branch is synced up with other remote branches, while the latest commits and code changes in your local repo are not affected. You are done!

# Updates 11/05/2020 (About elastic search indexing)

1. To query all the documents in the index "users" for recording user behaviors (on server side), run:

```
http localhost:9200/users/_search
```
2. To query all the documents in the index "regs" for registered user info (on server side), run:

```
http localhost:9200/regs/_search
```

