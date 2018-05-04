import csv
import os
import geoio
import sys
from osgeo import gdal

YEARS = ['2016', '2017']
IMGS = ['B3.TIF', 'B4.TIF', 'B5.TIF', 'B6.TIF', 'B7.TIF', 'B10.TIF', 'B11.TIF']
BANDS = ['G', 'R', 'NIR', 'SIR1', 'SIR2', 'TIR1', 'TIR2']

def findPath():
    p = os.path.abspath(__file__).split("/")
    p = p[:-2]
    fp = "/".join(p)
    return fp

def fileList(PATH):
    fl = list()
    for f in os.listdir(PATH):
        if f.endswith('.csv'):
            fl.append(f)
    return fl

def ndvi(NIR, RED):
    a = float(NIR)
    b = float(RED)
    return float((a-b)/(a+b))

def ndwiGao(NIR, SIR):
    a = float(NIR)
    b = float(SIR)
    return float((a-b)/(a+b))

def ndwiMcFeeters(NIR, GREEN):
    a = float(NIR)
    b = float(GREEN)
    return float((a-b)/(a+b))

def yieldData(DATAPATH):
    for y in YEARS:
        data = dict()
        rows = list()
        print "BUILDING YEAR: " + y
        # GET CSV DATA
        p = DATAPATH + '/csv/' + y
        for f in fileList(p):
            with open(p + '/' + f, 'rb') as rfp:
                reader = csv.DictReader(rfp)
                rows = rows + [r for r in reader]
        # GET PIXEL COORDS
        q = DATAPATH + '/img/' + y
        t = q + '/B3.TIF'
        pixelmap = dict()
        # lldict = dict()
        base = geoio.GeoImage(t)
        j = 0.0
        for r in rows:
            sys.stdout.write('BUILDING PIXELMAP:\t%.2f%s\r' % (float(j/len(rows))*100, "%"))
            sys.stdout.flush()
            j += 1.0
            xp, yp = base.proj_to_raster(float(r['Longitude']), float(r['Latitude']))
            k = (int(xp), int(yp))
            # lldict[km] = k
            if k not in pixelmap.keys():
                pixelmap[k] = {
                    'lat': float(r['Latitude']),
                    'lon': float(r['Longitude']),
                    'xp': float(xp),
                    'yp': float(yp),
                    'Product': list(),
                    'Yld Vol(Dry)(bu/ac)': 0.0,
                    'Count': 0,
                    'Year': int(y)
                }
            if r['Product'] not in pixelmap[k]['Product']:
                pixelmap[k]['Product'].append(r['Product'])
            pixelmap[k]['Yld Vol(Dry)(bu/ac)'] += float(r['Yld Vol(Dry)(bu/ac)'])
            pixelmap[k]['Count'] += 1
        sys.stdout.write('\n')
        for key in pixelmap:
            data[key] = dict(pixelmap[key])
        for i in range(0, len(IMGS)):
            sys.stdout.write('GATHERING:\t%s\r' % IMGS[i])
            sys.stdout.flush()
            src = gdal.Open(q + '/' + IMGS[i])
            band = src.GetRasterBand(1)
            cols = src.RasterXSize
            rows = src.RasterYSize
            srcDATA = band.ReadAsArray(0, 0, cols, rows)
            for key in data:
                data[key][BANDS[i]] = int(srcDATA[key[0]][key[1]])

        sys.stdout.write('\n')
        for k in data:
            data[k]['NDVI'] = ndvi(data[k]['NIR'], data[k]['R'])
            data[k]['NDWI1'] = ndwiGao(data[k]['NIR'], data[k]['SIR2'])
            data[k]['NDWI2'] = ndwiMcFeeters(data[k]['NIR'], data[k]['G'])
        # SAVE TO FILE
        print "SAVING TO FILE..."
        with open(DATAPATH + '/' + y + '.csv', 'wb') as wfp:
            headers = ['lon', 'lat', 'xp', 'yp', 'Product',
             'Yld Vol(Dry)(bu/ac)', 'Count', 'Year'] + BANDS + ['NDVI',
              'NDWI2', 'NDWI1']
            writer = csv.DictWriter(wfp, headers)
            writer.writeheader()
            for key in data.keys():
                writer.writerow(data[key])

def setup():
    DATAPATH = findPath() + '/data/proc'
    yieldData(DATAPATH)

def main():
    DATAPATH = findPath() + '/data/proc'
    yieldData(DATAPATH)

# =========================================

if __name__ == '__main__':
    main()
