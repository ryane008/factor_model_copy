# factor-modeling

## Using InfluxDB
- InfluxDB is the stack in which our factor data is stored.
- To start it, run influxdb
- To create  

### Installation
#### MacOS
- Download instructions can be found at the following link: https://docs.influxdata.com/influxdb/v2/install/#download-install-and-configure-the-influx-cli
1. Brew install influxdb
2. influxd 
    - To access the UI frontend for the DBMS, go to: http://localhost:8086
3. InfluxDB CLI (which allows you to use the command line for influxDB instructions) should be installed with brew versions 2.7 or after, but refer to the following instructions if not: https://docs.influxdata.com/influxdb/v2/tools/influx-cli/
    - You can check to see if InfluxDB CLI is installed with the following command: brew list | grep influxdb-cli

### Writing a CSV to InfluxDB

