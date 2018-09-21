# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:02:48 2017

@author: Toby

Script to replace numeric codes in road acident data files with real values
and output new files containing these values
"""

import pandas as pd
import numpy as np

######################## Load data ###########################################
acc16 = pd.read_csv('Acc_2016.csv')
cas16 = pd.read_csv('Cas_2016.csv')
veh16 = pd.read_csv('Veh_2016.csv')
refDataDict = pd.read_excel('Road-Accident-Safety-Data-Guide.xls',sheetname=None)

#################### Translate Data ##########################################


# Reference data codes
pf = refDataDict['Police Force'].set_index('code')
acc16['Police_Force'] = acc16['Police_Force'].apply(lambda x: pf.loc[x,'label'])

accSev = refDataDict['Accident Severity'].set_index('code')
acc16['Accident_Severity'] = acc16['Accident_Severity'].apply(lambda x: accSev.loc[x,'label'])

DoW = refDataDict['Day of Week'].set_index('code')
acc16['Day_of_Week'] = acc16['Day_of_Week'].apply(lambda x: DoW.loc[x,'label'])

LocAuthD = refDataDict['Local Authority (District)'].set_index('code')
acc16['Local_Authority_(District)'] = acc16['Local_Authority_(District)'].apply(lambda x: LocAuthD.loc[x,'label'])

LocAuthH = refDataDict['Local Authority (Highway)'].set_index('code')
acc16['Local_Authority_(Highway)'] = acc16['Local_Authority_(Highway)'].apply(lambda x: LocAuthH.loc[x,'label'])

RC1 = refDataDict['1st Road Class'].set_index('code')
acc16['1st_Road_Class'] = acc16['1st_Road_Class'].apply(lambda x: RC1.loc[x,'label'])

RT = refDataDict['Road Type'].set_index('code')
acc16['Road_Type'] = acc16['Road_Type'].apply(lambda x: RT.loc[x,'label'])

JD = refDataDict['Junction Detail'].set_index('code')
acc16['Junction_Detail'] = acc16['Junction_Detail'].apply(lambda x: JD.loc[x,'label'])

JC = refDataDict['Junction Control'].set_index('code')
acc16['Junction_Control'] = acc16['Junction_Control'].apply(lambda x: JC.loc[x,'label'])

acc16['2nd_Road_Class'].replace(-1,0,inplace=True) # Data does not match supplied data dictionary
RC2 = refDataDict['2nd Road Class'].set_index('code')
acc16['2nd_Road_Class'] = acc16['2nd_Road_Class'].apply(lambda x: RC2.loc[x,'label'])

PCH = refDataDict['Ped Cross - Human'].set_index('code')
acc16['Pedestrian_Crossing-Human_Control'] = acc16['Pedestrian_Crossing-Human_Control'].apply(lambda x: PCH.loc[x,'label'])

PCPh = refDataDict['Ped Cross - Physical'].set_index('code')
acc16['Pedestrian_Crossing-Physical_Facilities'] = acc16['Pedestrian_Crossing-Physical_Facilities'].apply(lambda x: PCPh.loc[x,'label'])

LC = refDataDict['Light Conditions'].set_index('code')
acc16['Light_Conditions'] = acc16['Light_Conditions'].apply(lambda x: LC.loc[x,'label'])

Weather = refDataDict['Weather'].set_index('code')
acc16['Weather_Conditions'] = acc16['Weather_Conditions'].apply(lambda x: Weather.loc[x,'label'])

RS = refDataDict['Road Surface'].set_index('code')
acc16['Road_Surface_Conditions'] = acc16['Road_Surface_Conditions'].apply(lambda x: RS.loc[x,'label'])

SCaS = refDataDict['Special Conditions at Site'].set_index('code')
acc16['Special_Conditions_at_Site'] = acc16['Special_Conditions_at_Site'].apply(lambda x: SCaS.loc[x,'label'])

CarHz = refDataDict['Carriageway Hazards'].set_index('code')
acc16['Carriageway_Hazards'] = acc16['Carriageway_Hazards'].apply(lambda x: CarHz.loc[x,'label'])

UrbRur = refDataDict['Urban Rural'].set_index('code')
acc16['Urban_or_Rural_Area'] = acc16['Urban_or_Rural_Area'].apply(lambda x: UrbRur.loc[x,'label'])

CasClass = refDataDict['Casualty Class'].set_index('code')
cas16['Casualty_Class'] = cas16['Casualty_Class'].apply(lambda x: CasClass.loc[x,'label'])

CasSex = refDataDict['Sex of Casualty'].set_index('code')
cas16['Sex_of_Casualty'] = cas16['Sex_of_Casualty'].apply(lambda x: CasSex.loc[x,'label'])

#Casualty Age Band to do

CasSev = refDataDict['Casualty Severity'].set_index('code')
cas16['Casualty_Severity'] = cas16['Casualty_Severity'].apply(lambda x: CasSev.loc[x,'label'])

PedLoc = refDataDict['Ped Location'].set_index('code')
cas16['Pedestrian_Location'] = cas16['Pedestrian_Location'].apply(lambda x: PedLoc.loc[x,'label'])

PedMov = refDataDict['Ped Movement'].set_index('code')
cas16['Pedestrian_Movement'] = cas16['Pedestrian_Movement'].apply(lambda x: PedMov.loc[x,'label'])

CarPas = refDataDict['Car Passenger'].set_index('code')
cas16['Car_Passenger'] = cas16['Car_Passenger'].apply(lambda x: CarPas.loc[x,'label'])

BusPas = refDataDict['Bus Passenger'].set_index('code')
cas16['Bus_or_Coach_Passenger'] = cas16['Bus_or_Coach_Passenger'].apply(lambda x: BusPas.loc[x,'label'])

PRMW = refDataDict['Ped Road Maintenance Worker'].set_index('code')
cas16['Pedestrian_Road_Maintenance_Worker'] = cas16['Pedestrian_Road_Maintenance_Worker'].apply(lambda x: PRMW.loc[x,'label'])

CasType = refDataDict['Casualty Type'].set_index('code')
cas16['Casualty_Type'] = cas16['Casualty_Type'].apply(lambda x: CasType.loc[x,'label'])

CasHA = refDataDict['Home Area Type'].set_index('code')
cas16['Casualty_Home_Area_Type'] = cas16['Casualty_Home_Area_Type'].apply(lambda x: CasHA.loc[x,'label'])

IMD = refDataDict['IMD Decile'].set_index('code')
cas16['Casualty_IMD_Decile'] = cas16['Casualty_IMD_Decile'].apply(lambda x: IMD.loc[x,'label'])

VehType = refDataDict['Vehicle Type'].set_index('code')
veh16['Vehicle_Type'] = veh16['Vehicle_Type'].apply(lambda x: VehType.loc[x,'label'])

TaA = refDataDict['Towing and Articulation'].set_index('code')
veh16['Towing_and_Articulation'] = veh16['Towing_and_Articulation'].apply(lambda x: TaA.loc[x,'label'])

VehMan = refDataDict['Vehicle Manoeuvre'].set_index('code')
veh16['Vehicle_Manoeuvre'] = veh16['Vehicle_Manoeuvre'].apply(lambda x: VehMan.loc[x,'label'])

VehLoc = refDataDict['Vehicle Location'].set_index('code')
veh16['Vehicle_Location-Restricted_Lane'] = veh16['Vehicle_Location-Restricted_Lane'].apply(lambda x: VehLoc.loc[x,'label'])

JunctLoc = refDataDict['Junction Location'].set_index('code')
veh16['Junction_Location'] = veh16['Junction_Location'].apply(lambda x: JunctLoc.loc[x,'label'])

Skidd = refDataDict['Skidding and Overturning'].set_index('code')
veh16['Skidding_and_Overturning'] = veh16['Skidding_and_Overturning'].apply(lambda x: Skidd.loc[x,'label'])

HOiC = refDataDict['Hit Object in Carriageway'].set_index('code')
veh16['Hit_Object_in_Carriageway'] = veh16['Hit_Object_in_Carriageway'].apply(lambda x: HOiC.loc[x,'label'])

HOoC = refDataDict['Hit Object Off Carriageway'].set_index('code')
veh16['Hit_Object_off_Carriageway'] = veh16['Hit_Object_off_Carriageway'].apply(lambda x: HOoC.loc[x,'label'])

VLC = refDataDict['Veh Leaving Carriageway'].set_index('code')
veh16['Vehicle_Leaving_Carriageway'] = veh16['Vehicle_Leaving_Carriageway'].apply(lambda x: VLC.loc[x,'label'])

POI = refDataDict['1st Point of Impact'].set_index('code')
veh16['1st_Point_of_Impact'] = veh16['1st_Point_of_Impact'].apply(lambda x: POI.loc[x,'label'])

LHD = refDataDict['Was Vehicle Left Hand Drive'].set_index('code')
veh16['Was_Vehicle_Left_Hand_Drive?'] = veh16['Was_Vehicle_Left_Hand_Drive?'].apply(lambda x: LHD.loc[x,'label'])

Purpose = refDataDict['Journey Purpose'].set_index('code')
veh16['Journey_Purpose_of_Driver'] = veh16['Journey_Purpose_of_Driver'].apply(lambda x: Purpose.loc[x,'label'])

DrSex = refDataDict['Sex of Driver'].set_index('code')
veh16['Sex_of_Driver'] = veh16['Sex_of_Driver'].apply(lambda x: DrSex.loc[x,'label'])

refDataDict['Vehicle Propulsion Code']['code'].replace(' M',-1,inplace=True) # Data does not match supplied data dictionary
Propulsion = refDataDict['Vehicle Propulsion Code'].set_index('code')
veh16['Propulsion_Code'] = veh16['Propulsion_Code'].apply(lambda x: Propulsion.loc[x,'label'])

#Age Band of Driver to do

veh16['Driver_IMD_Decile'] = veh16['Driver_IMD_Decile'].apply(lambda x: IMD.loc[x,'label'])

veh16['Driver_Home_Area_Type'] = veh16['Driver_Home_Area_Type'].apply(lambda x: CasHA.loc[x,'label'])

veh16['Vehicle_IMD_Decile'] = veh16['Vehicle_IMD_Decile'].apply(lambda x: IMD.loc[x,'label'])

acc16.to_csv('Acc_2016_Tidy.csv')
cas16.to_csv('Cas_2016_Tidy.csv')
veh16.to_csv('Veh_2016_Tidy.csv')   