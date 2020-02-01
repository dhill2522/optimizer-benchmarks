# Information about the data

## ERCOT grid data
The ercot grid data is compiled into a single SQLite database (`ercot_data.db`). It currently covers all the load and generation data from 2017 through 2019.

Right now the data is split up into two tables: `Generation` and `Load`. 

`Generation` contains all the generation data for the grid broken down by fuel source into 15-minute intervals. Below is a sample from the `Generation` table for a single 15-minute interval:
| Fuel | Generation | Date_Time | Resolution |
| ---- | ------     | -----     | -----      |
| Biomass   | 9.072768      | 2019-10-01 00:00:00.000000 | quarter-hour |
| Coal      | 1839.849139   | 2019-10-01 00:00:00.000000 | quarter-hour |
| Gas       | 831.853927    | 2019-10-01 00:00:00.000000 | quarter-hour |
| Gas-CC    | 4074.894757   | 2019-10-01 00:00:00.000000 | quarter-hour |
| Hydro     | 2.492367      | 2019-10-01 00:00:00.000000 | quarter-hour |
| Nuclear   | 1241.955868   | 2019-10-01 00:00:00.000000 | quarter-hour |
| Other     | 2.253087      | 2019-10-01 00:00:00.000000 | quarter-hour |
| Solar     | 0.0           | 2019-10-01 00:00:00.000000 | quarter-hour |
| Wind      | 3598.037892   | 2019-10-01 00:00:00.000000 | quarter-hour |

`Load` contains all the load data broken down by region as well as total for 1-hour intervals. Below is a sample from the `Load` table:
| HourEnding | COAST | EAST | FWEST | NORTH | NCENT | SOUTH | SCENT | WEST | ERCOT | Resolution |
| ---       | ---   | ----  | ----  | ----  | ----  | ----  | ----  | ----  | ---- | ----       |
| 2019-10-01 00:00:00 | 13391.87 | 1614.03 | 3559.48 | 885.22 | 15739.59 | 4038.34 | 8077.59 | 1380.09 | 48686.22 | hour |
| 2019-10-01 01:00:00 | 12480.88 | 1496.47 | 3602.53 | 843.91 | 14597.13 | 3774.4 | 7567.68 | 1324.36 | 45687.37 | hour |
| 2019-10-01 02:00:00 | 11982.42 | 1393.32 | 3544.35 | 814.37 | 13890.05 | 3605.58 | 7199.54 | 1282.87 | 43712.51 | hour |

Note that as of right now one table is in quarter-hour resolution while the other is in 1-hour resolution. This means that the data will have to be scaled one way or another when using both in the same model.

## Accessing SQLite data
While SQLite may not be as familiar as Excel or CSV it has many useful features and is still quite easy to use. You can find a tutorial on SQLite select statements [here](https://www.sqlitetutorial.net/sqlite-select/). Select statements are all you are likely to need if you are just looking to use the data, so don't get too caught up in `INSERT`, `DELETE` and other SQLite statements.

Below is a simple example of how to query a SQLite database using a written out SQL query. 
```python
# Create the connection to the database
con = create_engine('sqlite:///data/ercot_data.db')

# Create the sql query
query = "SELECT * FROM Load WHERE HourEnding BETWEEN date('2019-10-01') AND date('2019-10-02')"

# Load the data from the database
data = pd.read_sql(query, con, parse_dates=['HourEnding'])
```

You could also read in the entire table into Pandas and then filter the data in Python if you prefer, but that will be slower. The `Generation` table has almost one million rows, so iterating through the whole data set is rarely a good idea.

You can also find and download a SQLite database browser that will let you browse through the data and test SQL queries. There are many different possiblities, so look around at what is available for your platform and try one. One of my current favorites is [DB Browser for SQLite](https://sqlitebrowser.org/). It is available for all major platforms. Feel free to ask if you have questions or run into difficulty with this.