'''

Description of the product from CMEMS website:

The biogeochemical hindcast for global ocean is produced at Mercator-Ocean (Toulouse. France). It provides 3D biogeochemical fields since year 1993 at 1/4 degree and on 75 vertical levels. It uses PISCES biogeochemical model (available on the NEMO modelling platform). No data assimilation in this product.

    Latest NEMO version (v3.6_STABLE)
    Forcings: FREEGLORYS2V4 ocean physics produced at Mercator-Ocean and ERA-Interim atmosphere produced at ECMWF at a daily frequency
    Outputs: Daily (chlorophyll. nitrate. phosphate. silicate. dissolved oxygen. primary production) and monthly (chlorophyll. nitrate. phosphate. silicate. dissolved oxygen. primary production. iron. phytoplankton in carbon) 3D mean fields interpolated on a standard regular grid in NetCDF format. The simulation is performed once and for all.
    Initial conditions: World Ocean Atlas 2013 for nitrate. phosphate. silicate and dissolved oxygen. GLODAPv2 for DIC and Alkalinity. and climatological model outputs for Iron and DOC
    Quality/Accuracy/Calibration information: See the related QuID

DOI (product): https://doi.org/10.48670/moi-00019

-- MORE INFO --
Product ID: GLOBAL_MULTIYEAR_BGC_001_029

time spand 31 Dec 1992 to 30 May 2024

1993-01-01, 00:00→2022-12-01, 00:00

Mass concentration of chlorophyll a in sea waterchl [mg/m3]
Mole concentration of dissolved iron in sea waterfe [mmol/m3]
Mole concentration of nitrate in sea waterno3 [mmol/m3]
Net primary production of biomass expressed as carbon per unit volume in sea waternppv [mg/m3/day]
Mole concentration of dissolved molecular oxygen in sea watero2 [mmol/m3]
Sea water ph reported on total scaleph
Mole concentration of phytoplankton expressed as carbon in sea waterphyc [mmol/m3]
Mole concentration of phosphate in sea waterpo4 [mmol/m3]
Mole concentration of silicate in sea watersi [mmol/m3]
Surface partial pressure of carbon dioxide in sea waterspco2 [Pa]

variables=["chl", "fe", "no3", "nppv", "o2", "ph", "phyc", "po4", "si", "spco2"],

'''

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import datetime
# Copernicus API
import copernicusmarine
# for map
from matplotlib.gridspec import GridSpec
from cartopy.crs import EqualEarth, PlateCarree
# For polygons
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union

# Parameters
INTEGRATE = True
DOWNLOAD = False
REMOTE = False
ZMAX = 100

## ---- Get CMEMS data ---- ##
if REMOTE:
    ## Set parameters
    data_request = {
    "dataset_id" : "cmems_mod_glo_bgc_my_0.25deg_P1M-m",
    "longitude" : [-77, -42], 
    "latitude" : [35, 62],
    "depth" : [0, 1000],
    "time" : ["1993-01-01", "2022-12-01"],
    "variables" : ["nppv"]
    }

    ## Load xarray dataset
    ds = copernicusmarine.open_dataset(
        dataset_id = data_request["dataset_id"],
        minimum_longitude = data_request["longitude"][0],
        maximum_longitude = data_request["longitude"][1],
        minimum_latitude = data_request["latitude"][0],
        maximum_latitude = data_request["latitude"][1],
        minimum_depth = data_request["depth"][0],
        maximum_depth = data_request["depth"][1],
        start_datetime = data_request["time"][0],
        end_datetime = data_request["time"][1],
        variables = data_request["variables"]
        )

elif DOWNLOAD:
    ## This is to download data:
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_bgc_my_0.25deg_P1M-m",
        variables=["nppv"],
        minimum_longitude=-75,
        maximum_longitude=-40,
        minimum_latitude=30,
        maximum_latitude=75,
        start_datetime="1993-01-01T00:00:00",
        end_datetime="2022-12-01T00:00:00",
        minimum_depth=0.5057600140571594,
        maximum_depth=0.5057600140571594,
        )

else:
    print('use local data')
    ds = xr.open_dataset('PP.nc')

    
#Print loaded dataset information
print(ds)

# get dimensions
lons = ds.longitude.values
lats = ds.latitude.values
depth = ds.depth.values

# Select depth range
ds = ds.where((ds.depth<=ZMAX), drop=True) # original one

# Integrate or mean
if INTEGRATE:
    ds = ds.integrate('depth') #mgC/m2/day
else:        
    ds = ds.mean('depth')*ZMAX #mgC/m2/day

# Annual Net
ds = ds.resample(time='YS').mean()*365/1000 #gC/m3/yr

# To dataArray
da = ds['nppv']

# Test
## dz = np.gradient(depth)
## ds['depth'] = dz
## da = ds['nppv']
## da = da*da.depth #gC/m2/yr
## da = da.sum('depth')
## da = da.resample(time='YS').sum()/1000 #gC/m2/yr



## ---- Get EPUs ---- ##
epus = pd.read_csv('~/github/AZMP-NL/utils/EPU_Polygon.csv')
epu_list = epus.EPU.unique()
# store in a dict
dict = {}
for epu in epu_list:
    xlon = epus.loc[epus.EPU==epu].Lon
    xlat = epus.loc[epus.EPU==epu].Lat
    tmp = {'lat' : xlat, 'lon' : xlon}
    dict[epu] = tmp
    del tmp, xlon, xlat

Labrador_Shelf =  Polygon(zip(dict['Labrador_Shelf']['lon'], dict['Labrador_Shelf']['lat']))
Newfoundland_Shelf = Polygon(zip(dict['Newfoundland_Shelf']['lon'], dict['Newfoundland_Shelf']['lat']))
Grand_Bank = Polygon(zip(dict['Grand_Bank']['lon'], dict['Grand_Bank']['lat']))
Flemish_Cap = Polygon(zip(dict['Flemish_Cap']['lon'], dict['Flemish_Cap']['lat']))
Southern_Newfoundland = Polygon(zip(dict['Southern_Newfoundland']['lon'], dict['Southern_Newfoundland']['lat']))
Scotian_Shelf = Polygon(zip(dict['Scotian_Shelf']['lon'], dict['Scotian_Shelf']['lat']))
Gulf_of_Maine = Polygon(zip(dict['Gulf_of_Maine']['lon'], dict['Gulf_of_Maine']['lat']))
Georges_Bank = Polygon(zip(dict['Georges_Bank']['lon'], dict['Georges_Bank']['lat']))
Mid_Atlantic_Bight = Polygon(zip(dict['Mid_Atlantic_Bight']['lon'], dict['Mid_Atlantic_Bight']['lat']))

## ---- Extract PP per polygon ---- ##
# Create mask
Labrador_Shelf_mask = np.zeros((len(lats), len(lons)))*np.nan
Newfoundland_Shelf_mask = np.zeros((len(lats), len(lons)))*np.nan
Grand_Bank_mask = np.zeros((len(lats), len(lons)))*np.nan
Flemish_Cap_mask = np.zeros((len(lats), len(lons)))*np.nan
Southern_Newfoundland_mask = np.zeros((len(lats), len(lons)))*np.nan       
Scotian_Shelf_mask = np.zeros((len(lats), len(lons)))*np.nan
Gulf_of_Maine_mask = np.zeros((len(lats), len(lons)))*np.nan            
Georges_Bank_mask = np.zeros((len(lats), len(lons)))*np.nan
Mid_Atlantic_Bight_mask = np.zeros((len(lats), len(lons)))*np.nan


# counter
Labrador_Shelf_counter = 0
Newfoundland_Shelf_counter = 0
Grand_Bank_counter = 0
Flemish_Cap_counter = 0
Southern_Newfoundland_counter = 0
Scotian_Shelf_counter = 0
Gulf_of_Maine_counter = 0            
Georges_Bank_counter = 0
Mid_Atlantic_Bight_counter = 0

# select data in polygon
for i, xx in enumerate(lons):
    for j, yy in enumerate(lats):
        point = Point(lons[i], lats[j])
        if Labrador_Shelf.contains(point):
            Labrador_Shelf_counter = Labrador_Shelf_counter + 1
            Labrador_Shelf_mask[j,i] = 0
        elif Newfoundland_Shelf.contains(point):
            Newfoundland_Shelf_counter = Newfoundland_Shelf_counter + 1
            Newfoundland_Shelf_mask[j,i] = 0
        elif Grand_Bank.contains(point):
            Grand_Bank_counter = Grand_Bank_counter + 1
            Grand_Bank_mask[j,i] = 0
        elif Flemish_Cap.contains(point):
            Flemish_Cap_counter = Flemish_Cap_counter + 1
            Flemish_Cap_mask[j,i] = 0
        elif Southern_Newfoundland.contains(point):
            Southern_Newfoundland_counter = Southern_Newfoundland_counter + 1
            Southern_Newfoundland_mask[j,i] = 0
        elif Scotian_Shelf.contains(point):
            Scotian_Shelf_counter = Scotian_Shelf_counter + 1
            Scotian_Shelf_mask[j,i] = 0
        elif Gulf_of_Maine.contains(point):
            Gulf_of_Maine_counter = Gulf_of_Maine_counter + 1
            Gulf_of_Maine_mask[j,i] = 0
        elif Georges_Bank.contains(point):
            Georges_Bank_counter = Georges_Bank_counter + 1
            Georges_Bank_mask[j,i] = 0
        elif Mid_Atlantic_Bight.contains(point):
            Mid_Atlantic_Bight_counter = Mid_Atlantic_Bight_counter + 1
            Mid_Atlantic_Bight_mask[j,i] = 0
            
y_Labrador_Shelf, x_Labrador_Shelf = np.where(Labrador_Shelf_mask==0)
y_Newfoundland_Shelf, x_Newfoundland_Shelf = np.where(Newfoundland_Shelf_mask==0)
y_Grand_Bank, x_Grand_Bank = np.where(Grand_Bank_mask==0)
y_Flemish_Cap, x_Flemish_Cap = np.where(Flemish_Cap_mask==0)
y_Southern_Newfoundland, x_Southern_Newfoundland = np.where(Southern_Newfoundland_mask==0)
y_Scotian_Shelf, x_Scotian_Shelf = np.where(Scotian_Shelf_mask==0)
y_Gulf_of_Maine, x_Gulf_of_Maine = np.where(Gulf_of_Maine_mask==0)
y_Georges_Bank, x_Georges_Bank = np.where(Georges_Bank_mask==0)
y_Mid_Atlantic_Bight, x_Mid_Atlantic_Bight = np.where(Mid_Atlantic_Bight_mask==0)

da_Labrador_Shelf = da.isel(longitude=x_Labrador_Shelf, latitude=y_Labrador_Shelf)
print(da_Labrador_Shelf.shape)
lon_Labrador_Shelf = da_Labrador_Shelf.longitude.values
lat_Labrador_Shelf = da_Labrador_Shelf.latitude.values
df_Labrador_Shelf = da_Labrador_Shelf.mean(('latitude', 'longitude')).to_pandas()
del da_Labrador_Shelf

da_Newfoundland_Shelf = da.isel(longitude=x_Newfoundland_Shelf, latitude=y_Newfoundland_Shelf)
print(da_Newfoundland_Shelf.shape)
lon_Newfoundland_Shelf = da_Newfoundland_Shelf.longitude.values
lat_Newfoundland_Shelf = da_Newfoundland_Shelf.latitude.values
df_Newfoundland_Shelf = da_Newfoundland_Shelf.mean(('latitude', 'longitude')).to_pandas()
del da_Newfoundland_Shelf

da_Grand_Bank = da.isel(longitude=x_Grand_Bank, latitude=y_Grand_Bank)
print(da_Grand_Bank.shape)
lon_Grand_Bank = da_Grand_Bank.longitude.values
lat_Grand_Bank = da_Grand_Bank.latitude.values
df_Grand_Bank = da_Grand_Bank.mean(('latitude', 'longitude')).to_pandas()
del da_Grand_Bank

da_Flemish_Cap = da.isel(longitude=x_Flemish_Cap, latitude=y_Flemish_Cap)
print(da_Flemish_Cap.shape)
lon_Flemish_Cap = da_Flemish_Cap.longitude.values
lat_Flemish_Cap = da_Flemish_Cap.latitude.values
df_Flemish_Cap = da_Flemish_Cap.mean(('latitude', 'longitude')).to_pandas()
del da_Flemish_Cap

da_Southern_Newfoundland = da.isel(longitude=x_Southern_Newfoundland, latitude=y_Southern_Newfoundland)
print(da_Southern_Newfoundland.shape)
lon_Southern_Newfoundland = da_Southern_Newfoundland.longitude.values
lat_Southern_Newfoundland = da_Southern_Newfoundland.latitude.values
df_Southern_Newfoundland = da_Southern_Newfoundland.mean(('latitude', 'longitude')).to_pandas()
del da_Southern_Newfoundland

da_Scotian_Shelf = da.isel(longitude=x_Scotian_Shelf, latitude=y_Scotian_Shelf)
print(da_Scotian_Shelf.shape)
lon_Scotian_Shelf = da_Scotian_Shelf.longitude.values
lat_Scotian_Shelf = da_Scotian_Shelf.latitude.values
df_Scotian_Shelf = da_Scotian_Shelf.mean(('latitude', 'longitude')).to_pandas()
del da_Scotian_Shelf

da_Gulf_of_Maine = da.isel(longitude=x_Gulf_of_Maine, latitude=y_Gulf_of_Maine)
print(da_Gulf_of_Maine.shape)
lon_Gulf_of_Maine = da_Gulf_of_Maine.longitude.values
lat_Gulf_of_Maine = da_Gulf_of_Maine.latitude.values
df_Gulf_of_Maine = da_Gulf_of_Maine.mean(('latitude', 'longitude')).to_pandas()
del da_Gulf_of_Maine

da_Georges_Bank = da.isel(longitude=x_Georges_Bank, latitude=y_Georges_Bank)
print(da_Georges_Bank.shape)
lon_Georges_Bank = da_Georges_Bank.longitude.values
lat_Georges_Bank = da_Georges_Bank.latitude.values
df_Georges_Bank = da_Georges_Bank.mean(('latitude', 'longitude')).to_pandas()
del da_Georges_Bank

da_Mid_Atlantic_Bight = da.isel(longitude=x_Mid_Atlantic_Bight, latitude=y_Mid_Atlantic_Bight)
print(da_Mid_Atlantic_Bight.shape)
lon_Mid_Atlantic_Bight = da_Mid_Atlantic_Bight.longitude.values
lat_Mid_Atlantic_Bight = da_Mid_Atlantic_Bight.latitude.values
df_Mid_Atlantic_Bight = da_Mid_Atlantic_Bight.mean(('latitude', 'longitude')).to_pandas()
del da_Mid_Atlantic_Bight


## ---- Concat and Save data ---- ##
# Concat
PP = pd.concat([df_Labrador_Shelf, df_Newfoundland_Shelf, df_Grand_Bank, df_Flemish_Cap, df_Southern_Newfoundland, df_Scotian_Shelf, df_Gulf_of_Maine, df_Georges_Bank, df_Mid_Atlantic_Bight], keys=['Labrador_Shelf', 'Newfoundland_Shelf', 'Grand_Bank', 'Flemish_Cap', 'Southern_Newfoundland', 'Scotian_Shelf', 'Gulf_of_Maine', 'Georges_Bank', 'Mid_Atlantic_Bight'], axis=1)
PP.index = PP.index.year

# Plot
PP.plot()
plt.grid()
if INTEGRATE:
    #plt.ylabel(r'Net Annual Primary production ($\rm mgC\,m^{-2}\,day^{-1}$)')
    #PP.to_csv('Net_PP_integrated_mgC.m-2.day-1.csv', float_format='%.3f')
    plt.ylabel(r'Net Annual Primary production ($\rm gC\,m^{-2}\,y^{-1}$)')
    PP.to_csv('Net_PP_integrated_gC.m-2.y-1_00019_model_noasimilation.csv', float_format='%.3f')
else:        
    plt.ylabel(r'Net Annual Primary production ($\rm gC\,m^{-2}\,y^{-1}$)')
    PP.to_csv('Net_PP_gC.m-2.y-1_00019_model_noasimilation.csv', float_format='%.3f')

