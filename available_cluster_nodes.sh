#!/usr/bin/env bash

function all_available_nodes() {
  function all_nodes() {
    sinfo --Node --noheader --format="%N" "$@" | sort
  }

  function unavailable_nodes() {
    all_nodes -R
  }

  comm -23 <(all_nodes) <(unavailable_nodes)
}


function available_cluster_nodes() {
    for cluster in "$@"; do
      for node in $(all_available_nodes | grep "${cluster}"); do
          echo $node
          # setup_docker_on_node "${node}"
      done
    done
}


available_cluster_nodes $@



