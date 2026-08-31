# Git Cheat Sheet

## Basic Commands

- `git clone <repository>` : Clone an existing repository
- `git status` : Show the working directory status
- `git add <file>` : Add a file to the staging area
- `git commit -m "message"` : Commit changes with a message
- `git push` : Push changes to the remote repository
- `git pull` : Pull changes from the remote repository
- `git branch` : List, create, or delete branches
- `git checkout <branch>` : Switch to a different branch
- `git merge <branch>` : Merge a branch into the current branch

## Branches

Branches are used to develop features, fix bugs, or safely experiment with new ideas in a contained area of your repository.
They are intended to isolate changes from the main codebase, allowing multiple contributors to work on different features or fixes simultaneously without interfering with each other's work.

Most branches should be very short-lived and focused on **one** specific task or feature. Once the task is complete, the branch should be merged back into the main branch to make the feature part of the main codebase.

- `git branch` : List all branches
- `git branch <branch>` : Create a new branch
- `git branch -d <branch>` : Delete a branch
- `git checkout <branch>` : Switch to a different branch
- `git merge <branch>` : Merge a branch into the current branch

## Workflow

We will be using a workflow referred to as the ["feature branch workflow"](https://gist.github.com/blackfalcon/8428401).
A typical Git workflow involves the following steps:

1. **Synchronize with the main branch of the remote repository**: `git checkout main` and `git pull origin main`; this ensures that you have the most recent changes from the main branch.
2. **Create a new branch**: `git branch <branch>` and `git checkout <branch>`; the name of the branch should reflect the feature or task you are working on.
3. Repeat the following steps until you are satisfied with your changes; don't forget to include unit tests and documentation as needed.
   
   - **Make changes**: Edit files in your working directory until you complete a meaningful subtask.
   - **Review changes**: `git status` and possibly `git diff`: Review the changes you have made to ensure they are correct and complete before staging them.
   - **Stage changes**: `git add <file>`: Add the changes you have made to the staging area, preparing them for a commit.
   - **Commit changes**: `git commit -m "message"`: Record the changes in the local repository with a descriptive message.
   - **Push changes**: `git push origin <branch>`
   - **Resolve any conflicts**: It is possible that the changes you are trying to push conflict with changes made by others to your branch. If that happens, try `git pull origin <branch>` to fetch and merge the latest changes from the remote branch, and then resolve any conflicts that arise before pushing again.

5. **Create a pull request**: Go to the Github repository in your browser and create a pull request for the branch you have been working on. This will initiate a review process where other contributors can provide feedback and approve the changes before they are merged into the main branch.
6. **Merge changes**: The Github GUI on the web will provide an option to merge the pull request once it has been approved. Click the "Merge" button to integrate the changes into the main branch.
7. **Delete the branch** (optional): Deleting the now-completed branch is an option in the Github GUI.
