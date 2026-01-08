#!/bin/bash

set -e

# Create virtual interfaces
function create_virtual_interfaces {
    count=0
    base_ip="172.25.0"
    gateway_ip="172.25.0.1"

    clients_conditions=(
        "1 5mbit 10ms 0.1% veth1 veth-peer1"
        "2 5mbit 10ms 0.1% veth2 veth-peer2"
        "3 5mbit 10ms 0.1% veth3 veth-peer3"
        "4 5mbit 10ms 0.1% veth4 veth-peer4"
        "5 5mbit 10ms 0.1% veth5 veth-peer5"
        "6 5mbit 10ms 0.1% veth6 veth-peer6"
        "7 2mbit 50ms 1% veth7 veth-peer7"
        "8 2mbit 50ms 1% veth8 veth-peer8"
        "9 1mbit 100ms 2% veth9 veth-peer9"
        "10 1mbit 100ms 2% veth10 veth-peer10"
        "11 1mbit 100ms 2% veth11 veth-peer11"
        "12 1mbit 100ms 2% veth12 veth-peer12"
        "13 1mbit 100ms 2% veth13 veth-peer13"
        "14 1mbit 100ms 2% veth14 veth-peer14"
        "15 1mbit 100ms 2% veth15 veth-peer15"
        "16 1mbit 200ms 5% veth16 veth-peer16"
        "17 1mbit 200ms 5% veth17 veth-peer17"
        "18 1mbit 200ms 5% veth18 veth-peer18"
        "19 1mbit 200ms 5% veth19 veth-peer19"
        "20 500mbit 200ms 10% veth20 veth-peer20"
    )

    for condition in "${clients_conditions[@]}"; do
        count=$((count + 1))
        client_id=$(echo $condition | awk '{print $1}')
        bandwidth=$(echo $condition | awk '{print $2}')
        latency=$(echo $condition | awk '{print $3}')
        loss=$(echo $condition | awk '{print $4}')
        interface=$(echo $condition | awk '{print $5}')
        peer_interface=$(echo $condition | awk '{print $6}')
        traffic_flag=$(echo $condition | awk '{print $7}')

        echo "Creating virtual interface $interface and peer interface $peer_interface"
        ip link add $interface type veth peer name $peer_interface

        ip link set $interface up
        ip link set $peer_interface up

        ip_addr="${base_ip}.$((count + 10))/16"
        ip addr add $ip_addr dev $interface

        echo "Assigning IP $ip_addr to interface $interface"
        ip route del ${base_ip}.0/16 dev $interface
    done

    # Configure the default route to use eth0
    ip route add default via $gateway_ip dev eth0
}

create_virtual_interfaces

# Execute the original entrypoint if any
exec "$@"
