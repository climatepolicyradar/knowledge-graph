# Infra

Starts up an EC2 instance for running wikibase

Get it running by running `pulumi up` from the `infra` directory. Run `pulumi destroy` to tear it down.


## SSH Key Generation (On Mac)

```shell
ssh-keygen -t rsa -b 2048 -C "YOUR_USER_NAME@climatepolicyradar.org" -f ~/.ssh/wikibase
```

## Create a new EC2 instance

```shell
pulumi up
```

## Export the pulumi stack outputs to the shell

```shell
export $(pulumi stack output --shell)
```

## Connect to the instance in the terminal

```shell
sh -i $PRIVATE_KEY ubuntu@$EC2_PUBLIC_DNS -L 3000:localhost:3000
```

This will forward port 3000 on the EC2 instance to port 3000 on your local machine.

## Connect to the instance in VSCode

```shell
code --remote ssh-remote+ubuntu@$EC2_PUBLIC_DNS
```
