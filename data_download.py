#!/home/meichen/anaconda3/bin/python

def mk_path(path):

##**************************************##

# This function delete the original directory and make a new one.

##**************************************##
    import os
    import subprocess
    isExist = os.path.exists(path)
    if isExist:
        subprocess.call(['rm -r {}'.format(path)],shell=True)
    os.makedirs(path)

def mk_path_exist(path):

##**************************************##

# This function make a new directory if it does not exist.

##**************************************##
    import os
    import subprocess
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

def download_data(**kwargs):

##**************************************##
# Download data through fdsn
# Original download data is miniseed and stationxml files, then change to 
# standard SAC files. Channels with gaps and coverage smaller than 95% of desired
# time range are excluded. Minimum interstation distance is 100 meters.

# Created by Meichen Liu on June 8th, 2019

##**************************************##

## parameters
# event_info	in array, [event time, event latitude, event longitude]
# shape		shape of station domain
#		rec	rectangular domain
#		cir	circular domain
# minlat	minimum latitude if shape=rec
# maxlat	maximum latitude if shape=rec
# minlon	minimum longitude if shape=rec
# maxlon	maximum longitude if shape=rec
# cenlat	central latitude if shape=cir
# cenlon	central longitude if shape=cir
# minrad	minimum radius in degrees if shape=cir
# maxrad	maximum radius in degrees if shape=cir
# start		seconds before event time
# end		seconds after event time
# channel	the channel choose for download,"BH[NEZ]","HH[NEZ]",see IRIS 
#		classification
# network	the network code
# client	All currently availabel web service providers are:
#		"BGR", "EMSC", "ETH", "GEONET", "GFZ", "ICGC", "INGV", "IPGP",
#		"IRIS", "ISC", "KNMI", "KOERI", "LMU", "NCEDC", "NIEP", "NOA",
#		"ODC", "ORFEUS", "RESIF", "SCEDC", "TEXNET", "USGS", "USP"
# save_path	the path to save files including miniseed, stationxml, and SAC files.

    import obspy
    import numpy as np
    from obspy.clients.fdsn.mass_downloader import CircularDomain, RectangularDomain, Restrictions, MassDownloader
    from sys import argv

    event_info = kwargs.get('event_info')
    shape = kwargs.get('shape')
    if shape == 'rec':
        minlat = kwargs.get('minlat')
        maxlat = kwargs.get('maxlat')
        minlon = kwargs.get('minlon')
        maxlon = kwargs.get('maxlon')
    elif shape == 'cir':
        cenlat = kwargs.get('cenlat')
        cenlon = kwargs.get('cenlon')
        minrad = kwargs.get('minrad')
        maxrad = kwargs.get('maxrad')
    start = kwargs.get('start')
    end = kwargs.get('end')
    channel = kwargs.get('channel')
    client = kwargs.get('client')
    save_path = kwargs.get('save_path')
    mk_path(save_path)
    network = ['CI']

    origin_time = obspy.UTCDateTime(event_info[0])
    event_lat = event_info[1]
    event_lon = event_info[2]
    if shape == 'rec':
        domain = RectangularDomain(minlatitude=minlat,maxlatitude=maxlat,minlongitude=minlon,maxlongitude=maxlon)
    elif shape == 'cir':
        domain = CircularDomain(latitude=cenlat,longitude=cenlon,minradius=minrad,maxradius=maxrad)
   
    restrictions = Restrictions(starttime=origin_time-start,endtime=origin_time+end,reject_channels_with_gaps=True,minimum_length=0.95,minimum_interstation_distance_in_m=1E2,network='CI',channel_priorities=["{}".format(channel)],location_priorities=["","00","10","20","01","02"])
    mdl=MassDownloader(providers=['{}'.format(client)])
    mdl.download(domain,restrictions,mseed_storage="{}/waveforms".format(save_path),stationxml_storage="{}/stations".format(save_path))

def mseed2sac(**kwargs):

##**************************************##

# This function transform miniseed files to SAC files.
# First the javascript "stationxml-seed-converter-2.0.0.jar" produces dataless 
# files from stationxml files. Then "rdseed" using dataless files to tranform 
# mseed to SAC. Make sure rdseed is installed. Note that SAC files created from
# mseed lack some event information including event latitude, event longitude, 
# and event depth.

# Created by Meichen Liu on June 8th, 2019

##**************************************##

##parameters
# save_path	same as save_path in the function download_data
# jar_path	where the script "stationxml-seed-converter-2.0.0.jar is stored

    import subprocess
    import glob
    import os

    save_path = kwargs.get('save_path')
    jar_path = kwargs.get('jar_path')
    os.chdir('{}'.format(save_path))
    mk_path('{}/waveforms/SAC_files'.format(save_path))
    # change directory to save_path because rdseed does not work with home directory in the path

    for stnxml in glob.glob('{}/stations/*'.format(save_path)):
        stationname = stnxml.split('/')[-1]
        nw = stationname.split('.')[0]
        stn = stationname.split('.')[1]
        subprocess.call(['java','-jar','{}/stationxml-seed-converter-2.0.4-SNAPSHOT.jar'.format(jar_path),'--input','{}/stations/{}.{}.xml'.format(save_path,nw,stn),'--output','{}/waveforms/{}.{}.dataless'.format(save_path,nw,stn)])


        for filename in glob.glob('{}/waveforms/{}.{}.*.mseed'.format(save_path,nw,stn)):
            mseedfile = filename.split('/')[-1]
            subprocess.call(['rdseed','-pRdf','waveforms/{}'.format(mseedfile),'-z','1','-g','waveforms/{}.{}.dataless'.format(nw,stn),'-q','waveforms/SAC_files'])

def rename(**kwargs):

##**************************************##

# This function rename sac files to simpler, clearer filenames.

# Created by Meichen Liu on June 12nd, 2019 based on sac-manualv3.6

##**************************************##

##parameters
# dirname	the directory SAC files are saved

    import os
    import sys
    import glob

    dirname = kwargs.get('dirname')
    eventid = kwargs.get('eventid')

    os.putenv("SAC_DISPLAY_COPYRIGHT","O")
    os.chdir('{}'.format(dirname))

    for filename in glob.glob("*.SAC"):
        nw, stn, loc, chn = filename.split('.')[6:10]
        os.rename(filename,"%s.%s.%s.%s.%s.SAC.raw" % (eventid,nw, stn, loc, chn))


def add_info(**kwargs):

##**************************************##

# This function add event info to SAC files, including event latitude, event
# longitude, event depth, event magnitude, event original time.

# Created by Meichen Liu on June 10th, 2019 based on sac-manual-v3.6
##**************************************##

##parameters
# filename	the name of the SAC file
# dirname	the directory where SAC files are saved
# origin	the original time of the event
# evlo		the longitude of the event
# evla		the latitude of the event
# mag		the magnitude of the event

    import os
    import sys
    import datetime
    import subprocess
    import glob
    import numpy as np

    filename = kwargs.get('filename')
    dirname = kwargs.get('dirname')
    origin = kwargs.get('origin')
    evlo = kwargs.get('evlo')
    evla = kwargs.get('evla')
    evdp = kwargs.get('evdp')
    mag = kwargs.get('mag')

    os.putenv("SAC_DISPLAY_COPYRIGHT","O")
    os.chdir('{}'.format(dirname))

    o = datetime.datetime.strptime(origin,'%Y-%m-%dT%H:%M:%S')
    # calculate which day in a year is the occurence date
    jday = o.strftime("%j")

    p = subprocess.Popen(['sac'],stdin=subprocess.PIPE)
    s = "wild echo off \n"

    filelist = glob.glob('{}'.format(filename))
    if len(filelist)>20:
        for i in np.arange(round(len(filelist)/20)):
            s += "r %s \n" % filelist[i*20:i*20+20]
            s += "synchronize \n"
            s += "ch o gmt %s %s %s %s %s \n" % (o.year, jday, o.hour, o.minute, o.second)
            s += "ch allt (0 - &1,o&) iztype IO \n"
            s += "ch evlo %s evla %s evdp %s mag %s \n" % (evlo, evla, evdp, mag)
            s += "wh \n"
        s += "r %s \n" % filelist[i*20+20::]
        s += "synchronize \n"
        s += "ch o gmt %s %s %s %s %s \n" % (o.year, jday, o.hour, o.minute, o.second)
        s += "ch allt (0 - &1,o&) iztype IO \n"
        s += "ch evlo %s evla %s evdp %s mag %s \n" % (evlo, evla, evdp, mag)
        s += "wh \n"
    else:
        s += "r %s \n" % filename
        s += "synchronize \n"
        s += "ch o gmt %s %s %s %s %s \n" % (o.year, jday, o.hour, o.minute, o.second)
        s += "ch allt (0 - &1,o&) iztype IO \n"
        s += "ch evlo %s evla %s evdp %s mag %s \n" % (evlo, evla, evdp, mag)
        s += "wh \n"
    s += "q \n"
    p.communicate(s.encode())
    

def rm_response(**kwargs):

##**************************************##

# This function remove instrument response from seismograms directly created
# from SEED or miniSEED, including remean, retrend and taper. All processes are
# executed in sac.

# Created by Meichen Liu on June 10th, 2019 based on sac-manual-v3.6

##**************************************##

##parameters
# filename	the name of file to remove instrument response
# dirname	The directory where SAC files to be processed are saved
# f1-f4		Frequency limits. Lowpass and highpass to surpress low and high
#		frequencies. The four numbers should satisfy f1<f2<f3<f4. f4 
#		should be smaller than Nyquist frequency (1/2 sampling
#		frequency). f3 cannot be too close to f4. The distance between
#		f2 and f3 should be as large as possible. 
# ofile		the appendix of output filename

    import os
    import sys
    import glob
    import subprocess

    filename = kwargs.get('filename')
    dirname = kwargs.get('dirname')
    f1 = kwargs.get('f1')
    f2 = kwargs.get('f2')
    f3 = kwargs.get('f3')
    f4 = kwargs.get('f4')
    ofile = kwargs.get('ofile')

    os.chdir('{}'.format(dirname))
    os.putenv("SAC_DISPLAY_COPYRIGHT","O")

    p = subprocess.Popen(['sac'],stdin=subprocess.PIPE)
    s = "wild echo off \n"

    for sacfile in glob.glob('{}'.format(filename)):
        eventid,nw,stn,loc,chn,sac,seis_type = sacfile.split('.')
        pz = glob.glob("SAC_PZs_%s_%s_%s_%s_*" % (nw,stn,chn,loc))
        # multi PZ files are not considered
        if len(pz) != 1:
            print("PZ file error for %s" % sacfile)
        else:
            s += "r %s \n" % sacfile
            s += "rmean; rtr; taper \n"
            s += "trans from pol s %s to vel freq %s %s %s %s\n" % (pz[0],f1,f2,f3,f4)
            s += "mul 1.0e9 \n"
            s += "w %s.%s.%s.%s.%s.%s.%s \n" % (eventid,nw,stn,loc,chn,sac,ofile)

    s += "q \n"
    p.communicate(s.encode())

def rotate(**kwargs):

##**************************************##

# This function rotate NEZ components to RTZ components using sac. To rotate,
# each component of NEZ should exits with same delta. Headers should contain 
# STLA, STLO, EVLA, EVLO, which are necessary for GCP rotation. They should 
# have the same kzdata and kztime. And horizontal components also need to be 
# orthogonal to each other. Rotation would be successful if any of the above is
# not met.

# Created by Meichen Liu on June 10th. 2019 based on sac-manual-v3.6

##**************************************##

##parameters
# dirname	The directory where SAC files are saved.
# filename	the name of SAC file to be rotate
    
    import os
    import sys
    import glob
    import subprocess

    filename = kwargs.get('filename')
    dirname = kwargs.get('dirname')

    os.putenv("SAC_DISPLAY_COPYRIGHT","O")
    os.chdir('{}'.format(dirname))

    # create sets for each station of NEZ components
    sets = set()
    for sacfile in glob.glob("{}".format(filename)):
        eventid,nw, stn, loc, chn, sac, seis_type = sacfile.split('.')
        key = '.'.join([eventid,nw, stn, loc, chn[0:2]])
        sets.add(key)

    p = subprocess.Popen(['sac'], stdin=subprocess.PIPE)
    s = "wild echo off\n"
    for key in sets:
        Z = key + "Z.{}.{}".format(sac,seis_type)
        # if vertical component does not exist, loop to the next station. No
        # rotation would be done on this station.
        if not os.path.exists(Z):
            print("%s: Vertical component missing!" % key)
            continue

        # check if horizontal components exist
        if os.path.exists(key + "E.{}.{}".format(sac,seis_type)) and os.path.exists(key + "N.{}.{}".format(sac,seis_type)):
            E = key + "E.{}.{}".format(sac,seis_type)
            N = key + "N.{}.{}".format(sac,seis_type)
        elif os.path.exists(key + "1.{}.{}".format(sac,seis_type)) and os.path.exists(key + "2.{}.{}".format(sac,seis_type)):
            E = key + "E.{}.{}".format(sac,seis_type)
            N = key + "N.{}.{}".format(sac,seis_type)
        else:
            print("%s: Horizontal components missing!" % key)
            continue

        # check if horizontal components are orthogonal
        Ecmpaz = subprocess.check_output(['saclst','cmpaz','f',E]).decode().split()[1]
        Ncmpaz = subprocess.check_output(['saclst','cmpaz','f',N]).decode().split()[1]
        cmpaz_delta = abs(float(Ecmpaz) - float(Ncmpaz))
        if not (abs(cmpaz_delta-90)<=0.01 or abs(cmpaz_delta-270)<=0.01):
            print("%s: cmpaz1=%s, cmpaz2=%s are not orthogonal!" % (key, Ecmpaz, Ncmpaz))
            continue

        # check B, E, DELTA
        Zb, Ze, Zdelta = subprocess.check_output(['saclst','b','e','delta','f',Z]).decode().split()[1::]
        Eb, Ee, Edelta = subprocess.check_output(['saclst','b','e','delta','f',E]).decode().split()[1::]
        Nb, Ne, Ndelta = subprocess.check_output(['saclst','b','e','delta','f',N]).decode().split()[1::]
        
        if not (float(Zdelta) == float(Edelta) and float(Zdelta) == float(Ndelta)):
            print("%s: delta not equal!" % key)
            continue

        # get the max B and min E to be the data window
        begin = max(float(Zb), float(Eb), float(Nb))
        end = min(float(Ze), float(Ee), float(Ne))

        # output filename
        R, T, Z0 = key + 'R.{}.{}'.format(sac,seis_type), key + 'T.{}.{}'.format(sac,seis_type), key + 'Z.{}.{}'.format(sac,seis_type)

        s += "cut %f %f \n" % (begin, end)
        s += " r %s %s \n" % (E, N)
        s += "rotate to gcp \n"
        s += "w %s %s \n" % (R, T)
        s += "r %s \n" % Z
        s += "w %s \n" % Z0 
    s += "q \n"
    p.communicate(s.encode())

    # delete original files
#    for sacfile in glob.glob("*.BH[NEZ].SAC*"):
#        os.unlink(sacfile)

def remove_files(path):
    
    import os
    import subprocess

    os.chdir('{}'.format(path))
    subprocess.call(['rm *.BHR* *.BHZ.* *.BHE* *.BHN* RESP* SAC_PZ*'],shell=True)
    subprocess.call(['rm ../*.mseed'],shell=True)
    subprocess.call(['rm ../*.dataless'],shell=True)
    subprocess.call(['rm -r ../../stations'],shell=True)
    subprocess.call(['mv *.vel ~/work1/Attn_TOMO/seismograms'],shell=True)

def main():
    
    import numpy as np
    import pandas as pd
    import sys

    path = '/home/meichen/Research/Attn_TOMO'
    save_path = '/home/meichen/work1/Attn_TOMO'

    saveout = sys.stdout
    saveerr = sys.stderr
    f = open('stdout.log','w')
    sys.stderr = f
    sys.stdout = f

    # Read in events info
    events = pd.read_csv('{}/events_temp'.format(path),sep=',')
    for eventid in events['id']:
        folder_path = '{}/event_{}'.format(save_path,eventid)
        index       = list(events['id']).index(eventid)
        eventtime   = events['time'][index]
        eventlat    = events['latitude'][index]
        eventlon    = events['longitude'][index]
        eventdep    = events['depth'][index]
        eventmag    = events['mag'][index]
        print("Start process event_{}".format(eventid))
        # Download event miniseed files
        download_data(event_info=[eventtime,eventlat,eventlon],shape='cir',cenlat=eventlat,cenlon=eventlon,minrad=0,maxrad=70,start=0,end=1800,channel='BH[NEZ]',client='IRIS',save_path=folder_path)
        # Miniseed to SAC
        mseed2sac(save_path=folder_path,jar_path='/home/meichen/bin')
        # Rename SAC files as nw.stn.loc.chn.SAC.raw
        rename(dirname='{}/waveforms/SAC_files'.format(folder_path),eventid=eventid)
        # Add event info to SAC files
        add_info(filename='*.raw',dirname='{}/waveforms/SAC_files'.format(folder_path),origin=eventtime,evla=eventlat,evlo=eventlon,evdp=eventdep,mag=eventmag)
        # Remove instrument response
        rm_response(filename='*.raw',dirname='{}/waveforms/SAC_files'.format(folder_path),f1=0.004,f2=0.006,f3=4,f4=5,ofile='vel')
        # Rotate NEZ to RTZ
        rotate(filename='*.BH[NEZ].*.vel',dirname='{}/waveforms/SAC_files'.format(folder_path))
        # remove unnecessary files
        remove_files('{}/waveforms/SAC_files'.format(folder_path))

    sys.stdout = saveout
    sys.stderr = saveerr
    f.close()

main()
