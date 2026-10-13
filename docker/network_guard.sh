#!/bin/sh
set -eu
# This container shares ONLY the Compose namespace, never the host namespace.
iptables -F OUTPUT
iptables -P OUTPUT DROP
iptables -A OUTPUT -d 127.0.0.11 -j DROP
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
ip6tables -F OUTPUT
ip6tables -P OUTPUT DROP
ip6tables -A OUTPUT -o lo -j ACCEPT
ip6tables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
mkdir -p /netstate
# Existing marker is read-only even for root after dropping DAC capabilities.
# Replace it atomically using directory ownership, retaining read-only readers.
readlink /proc/self/ns/net > /netstate/namespace.next
chmod 444 /netstate/namespace.next
mv -f /netstate/namespace.next /netstate/namespace
chmod 444 /netstate/namespace
echo 'Compose namespace outbound traffic blocked; loopback and replies allowed.'
