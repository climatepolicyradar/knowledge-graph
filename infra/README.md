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

## To copy the current directory to the instance

```shell
tar czf - ./ | pv | ssh ec2-user@$EC2_PUBLIC_DNS 'mkdir -p ~/wikibase && tar xzf - -C ~/wikibase/'
```

- `tar czf - ./`: Creates a gzipped tar archive (tar czf) of the current directory (./) and outputs it to standard output (-).
- `| pv |`: Pipes the output of the tar command to the next command, via pv to monitor the progress of the data through a pipe.
- `ssh ec2-user@$EC2_PUBLIC_DNS`: Initiates an SSH connection to the remote machine.
- `mkdir -p ~/wikibase`: On the remote machine, creates a directory named wikibase in the home directory of the ec2-user.
- `'tar xzf - -C ~/'`: Still on the remote machine, extracts (tar xzf) the gzipped tar archive from standard input (-) and specifies the extraction directory (-C ~/, the home directory of the ec2-user).
