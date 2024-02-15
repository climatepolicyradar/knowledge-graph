import json
from pathlib import Path

import pulumi
import pulumi_aws as aws
from pulumi import Config

config = Config()
PROJECT = config.require("project")
PUBLIC_KEY_PATH = Path.home() / ".ssh" / "wikibase.pub"

with open(PUBLIC_KEY_PATH, "r") as key_file:
    public_key = key_file.read()

key_pair = aws.ec2.KeyPair(f"keypair-{PROJECT}", public_key=public_key)

sec_group = aws.ec2.SecurityGroup(
    f"securityGroup-{PROJECT}",
    description="Enable SSH, HTTP, and HTTPS ingress, and all egress",
    ingress=[
        {
            "protocol": "tcp",
            "from_port": 22,
            "to_port": 22,
            "cidr_blocks": ["0.0.0.0/0"],
        },
        {
            "protocol": "tcp",
            "from_port": 80,
            "to_port": 80,
            "cidr_blocks": ["0.0.0.0/0"],
        },
        {
            "protocol": "tcp",
            "from_port": 443,
            "to_port": 443,
            "cidr_blocks": ["0.0.0.0/0"],
        },
    ],
    egress=[
        {"protocol": "-1", "from_port": 0, "to_port": 0, "cidr_blocks": ["0.0.0.0/0"]}
    ],
)

with open("./install.sh", "r", encoding="utf-8") as script_file:
    user_data_script = script_file.read()

instance = aws.ec2.Instance(
    f"instance-{PROJECT}",
    instance_type="t3.large",
    ami="ami-0766b4b472db7e3b9",
    ebs_block_devices=[
        aws.ec2.InstanceEbsBlockDeviceArgs(
            device_name="/dev/sdh",
            volume_size=64,
        ),
    ],
    vpc_security_group_ids=[sec_group.id],
    key_name=key_pair.key_name,
    user_data=user_data_script,
)


pulumi.export("KEY_PAIR", key_pair.id)
pulumi.export("PRIVATE_KEY_PATH", PUBLIC_KEY_PATH.with_suffix("").as_posix())
pulumi.export("INSTANCE_ID", instance.id)
pulumi.export("PUBLIC_IP", instance.public_ip)
pulumi.export("EC2_PUBLIC_DNS", instance.public_dns)
