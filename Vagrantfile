# -*- mode: ruby -*-
# vi: set ft=ruby :

# adapted from:
#   https://technology.amis.nl/2018/05/21/rapidly-spinning-up-a-vm-with-ubuntu-and-docker-on-my-windows-machine-using-vagrant-and-virtualbox/

$installvim = <<-SCRIPT
apt-get update
apt-get -y install vim
SCRIPT

Vagrant.configure(2) do |config|
  # The most common configuration options are documented and commented below.
  # For a complete reference, please see the online documentation at
  # https://docs.vagrantup.com.

  # Every Vagrant development environment requires a box. You can search for
  # boxes at https://atlas.hashicorp.com/search.
  # config.vm.box = "centos/7"
  config.vm.box = "debian/jessie64"
  # config.vm.box = "ubuntu/xenial64"
  # config.vm.box = "hashicorp/precise64"
  # config.vm.box = "ubuntu/trusty64"

  config.vm.define "ellwood-glacier"

  # access a port on your host machine (via localhost) and have all data forwarded to a port on the guest machine.
  # config.vm.network "forwarded_port", guest: 9092, host: 9092

  # Create a private network, which allows host-only access to the machine
  # using a specific IP.
  config.vm.network "private_network", ip: "192.168.188.110"

  #define a larger than default (40GB) disksize
  config.disksize.size = '50GB'
  
  config.vm.provider "virtualbox" do |vb|
    vb.name = "ellwood-glacier" # virtualbox GUI name
    vb.memory = 4096
    vb.cpus = 1
    vb.customize ["modifyvm", :id, "--natdnshostresolver1", "on"]
    vb.customize ["modifyvm", :id, "--natdnsproxy1", "on"]
  end

  # set up auto-sync of files
  config.vm.synced_folder ".", "/vagrant", type: "rsync", rsync__exclude: ".git/"

  # install vim in the new VM:
  config.vm.provision "shell", inline: $installvim

  # set up Docker in the new VM:
  config.vm.provision :docker

  # install docker-compose into the VM and run the docker-compose.yml file (if it exists)
  # every time the  VM starts (https://github.com/leighmcculloch/vagrant-docker-compose)
  config.vm.provision :docker_compose, yml: "/vagrant/docker-compose.yml", run:"always"

end

