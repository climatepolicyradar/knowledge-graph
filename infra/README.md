# Infra

Infrastructure (defined in pulumi) for an EC2 instance where the wikibase stack can be developed.

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

## Load the private key into the ssh-agent

```shell
ssh-add ~/.ssh/wikibase
```

## Connect to the instance in the terminal

```shell
ssh ec2-user@$EC2_PUBLIC_DNS -i $PRIVATE_KEY_PATH
```

## Connect to the instance in VSCode

Get the hostname by running `echo ec2-user@$EC2_PUBLIC_DNS` and then hit CTRL+SHIFT+P and type `Remote-SSH: Connect to Host...` and paste the hostname.
