for VARIABLE in {1..30}
do
    python -m Src.run -cn config_GW &
    sleep 5
done
