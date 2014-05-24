#!/usr/bin/python
# -*- coding: iso-8859-1 -*-
# create by: Massimo Di Pierro<mdipierro@cs.depaul.edu>
# modified by mpraggs to work with Python 3
# license: GPL2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

LOGO ="""

 #######   ######  ########     ##     ## ######## #### ##        ######
##     ## ##    ## ##     ##    ##     ##    ##     ##  ##       ##    ##
##     ## ##       ##     ##    ##     ##    ##     ##  ##       ##
##     ## ##       ##     ##    ##     ##    ##     ##  ##        ######
##  ## ## ##       ##     ##    ##     ##    ##     ##  ##             ##
##    ##  ##    ## ##     ##    ##     ##    ##     ##  ##       ##    ##
 ##### ##  ######  ########      #######     ##    #### ########  ######
Created by Massimo Di Pierro - License GPL2 - all-to-all convertion utility
For the latest source and documentation: http://code.google.com/p/qcdutils/
"""

USAGE ="""
Usage:

    qcdutils_get.py [options] sources

Examples:

    qcdutils_get.py --test
    qcdutils_get.py --convert ildg gauge.cold.12x8x8x8
    qcdutils_get.py --convert mdp --float *.ildg
    qcdutils_get.py --convert split.mdp *.mdp

"""

##### imports #############################################################

import urllib
import hashlib
import os
import re
import sys
import time
import datetime
import optparse
import struct
import mmap
import glob
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
import array
import fcntl
import logging
import traceback
import shelve
import xml.dom.minidom as dom
import xml.parsers.expat as expat
try:
    import termios
    import signal
    HAVE_PROGRESSBAR = True
except ImportError:
    HAVE_PROGRESSBAR = False


##### global variables #############################################################

CATALOG = 'qcdutils.catalog'
NOW = datetime.datetime.now()
MAXBYTES = 1000  # max number of bytes for buffered reading
PRECISION = {'f':32,'d':64}
(X,Y,Z,T) = (1,2,3,0) # the MDP index convetion, used intenrnally

def notify(*a):
    """
    just an alies for the print function, in case we need to move to Python 3.x
    """
    print(' '.join([str(x) for x in a]))

##### class Lime #############################################################

class Lime(object):
    """
    based on this: http://usqcd.jlab.org/usqcd-docs/c-lime/lime_1p2.pdf
    Lime is basically a poor man's version of TAR:
    it concatenates names byte strings.
    """
    @staticmethod
    def xml_parser(data):
        def f(name,dxml = dom.parseString(data)):
            return dxml.getElementsByTagName(name)[0].childNodes[0].nodeValue
        return f
    def __init__(self,filename,mode,version = 1):
        """
        >>> lime = Lime('filename','r' or 'w')
        """
        self.magic = 1164413355
        self.version = version
        self.filename = filename
        self.mode = mode
        self.file = open(filename,mode)
        self.records = [] # [(name,position,size)]
        if mode == 'r' or mode == 'rb':
            while True:
                header = self.file.read(144)
                if not header: break
                magic, null, size, name = struct.unpack(b'!iiq128s',header)
                if magic != 1164413355:
                    raise IOError("not in lime format")
                ### the following line is VERY IMPORTANT
                # name contains (must contain) the C-style string zero
                # this is potentially a serious security vulnerability
                # of the LIME file format
                name = name[:name.find('\0')]
                position = self.file.tell()
                self.records.append((name,position,size)) # in bytes
                padding = (8 - (size % 8)) % 8
                self.file.seek(size+padding,1)
        # self.dump_info()

    def dump_info(self,filename=None):
        f = open(filename or (self.filename+'.fromlime.info'),'wb')
        f.write("LIME records:\n")
        for a,b,c in self:
            for a,b,c in self:
                f.write('- %s [%sbytes]\n' % (a,c))
                if c<1000:
                    f.write('\n'+b.read(c)+'\n\n')

    def read(self,record):
        """
        reads a Lime record
        >>> lime = Lime('filename','r')
        >>> name, reader, size = lime.read(records = 0)
        """
        if not self.mode in ('r','rb'):
            raise RuntimeError("not suported")
        (name,position,size) = self.records[record]
        self.file.seek(position)
        return (name, self.file, size)
    def __iter__(self):
        """
        >>> lime = Lime('filename','r')
        >>> for name, reader, size in lime:
        >>> print name, size, reader.read(size)
        """
        for record in range(len(self)):
            yield self.read(record)
    def write(self,name,reader,size = None,chunk = MAXBYTES):
        """
        write a Lime record
        >>> lime = Lime('filename','w')
        >>> lime.write('record name','data',size = 4)
        data can be a string or a file object
        """
        if not self.mode in ('w','wb'):
            raise RuntimeError("not supported")
        if isinstance(reader,str):
            if size == None:
                size = len(reader)
            reader = StringIO(reader)
        # write record header
        position = self.file.tell()
        header = struct.pack(b'!iHHq128s',self.magic,self.version,0,size,name)
        self.file.write(header)
        # read data from reader and write to file
        if hasattr(reader,'read'):
            for i in xrange(size // chunk):
                data = reader.read(chunk)
                if len(data) != chunk:
                    raise IOError
                self.file.write(data)
            chunk = size % chunk
            data = reader.read(chunk)
            if len(data) != chunk:
                raise IOError
            self.file.write(data)
        else:
            for data in reader:
                self.file.write(data)
        # add padding bytes
        padding = (8 - (size % 8)) % 8
        self.file.write('\0'*padding)
        self.records.append((name,size,position))
    def close(self):
        self.file.close()
    def __len__(self):
        """
        returns the number of lime records
        """
        return len(self.records)
    def keys(self):
        """
        returns the name of lime records
        """
        return [name for (name,position,size) in self.records]

def test_lime():
    notify('making a dummy LIME file and writing junk in it...')
    lime = Lime('test.zzz.0.lime','w')
    lime.write('record1','01234567')
    lime.write('record2','012345678')
    file = cStringIO.StringIO('0123456789') # memory file
    lime.write('record3',file,10) # write file content as record
    lime.close()

    notify('reading the file back...')
    lime = Lime('test.zzz.0.lime','r')
    notify('file contans %s records' % len(lime))
    notify('they have names: %s' % lime.keys())

    for name,reader,size in lime:
        notify('record name: %s\nrecord size: %s\nrecord data: %s' % \
                   (name, size, reader.read(size)))
    lime.close()


def copy_lime(input,output):
    lime_in = Lime(input,'rb')
    lime_out = Lime(output,'wb')
    for name,reader,size in lime_in:
        lime_out.write(name,reader,size)
    lime_in.close()
    lime_out.close()


##### reunitarize #############################################################

def reunitarize(s):
    (a1re, a1im, a2re, a2im, a3re, a3im, b1re, b1im, b2re, b2im, b3re, b3im) = s
    c1re = a2re * b3re - a2im * b3im - a3re * b2re + a3im * b2im
    c1im = -(a2re * b3im + a2im * b3re - a3re * b2im - a3im * b2re)
    c2re = a3re * b1re - a3im * b1im - a1re * b3re + a1im * b3im
    c2im = -(a3re * b1im + a3im * b1re - a1re * b3im - a1im * b3re)
    c3re = a1re * b2re - a1im * b2im - a2re * b1re + a2im * b1im
    c3im = -(a1re * b2im + a1im * b2re - a2re * b1im - a2im * b1re)
    return (a1re, a1im, a2re, a2im, a3re, a3im,
            b1re, b1im, b2re, b2im, b3re, b3im,
            c1re, c1im, c2re, c2im, c3re, c3im)


def check_unitarity(items,tolerance = 1.0):
    """
    this does not quite checks for unitarity, only that numbers are in [-1,+1] range
    """
    errors = [x for x in items if x<-tolerance or x>tolerance]
    if errors:
        raise RuntimeError("matrix is not unitary")

def reorder(data,order1,order2): # data are complex numbers
    """
    reorders a list as in the example:
    >>> assert ''.join(reorder('AABBCCDD',[X,Y,Z,T],[Z,Y,X,T])) == 'CCBBAADD'
    """
    k = len(data)      # 4*9*2
    m = len(order1)    # 4
    n = k // m            # 9*2
    items = [None]*k
    for i in range(k):
        items[n*order1[i // n]+i%n] = data[i]
    items = [items[n*order2[i // n]+i%n] for i in range(k)]
    return items

assert ''.join(reorder('AABBCCDD',[X,Y,Z,T],[Z,Y,X,T])) == 'CCBBAADD'

##### Field readers #############################################################

class QCDFormat(object):
    site_order = [T,Z,Y,X] ### always unused but for reference
    link_order = [X,Y,Z,T] ### this is the order of links at the site level
    is_gauge = True or False
    def unpack(self,data):
        """
        unpacks a string of bytes from file into a list of float/double numbers
        """
        if self.precision.lower() == 'f':
            n = len(data) // 4
        elif self.precision.lower() == 'd':
            n = len(data) // 8
        else:
            raise IOError("incorrect input precision")
        unpack_string = b'{}{}{}'.format(self.endianess, n,
                                         self.precision)
        items = struct.unpack(unpack_string, data)
        if self.is_gauge:
            items = reorder(items,self.link_order,(T,X,Y,Z))
            check_unitarity(items)
        return items
    def pack(self,items):
        """
        packs a list of float/double numbers into a string of bytes
        """
        if self.is_gauge:
            items = reorder(items,(T,X,Y,Z),self.link_order)
        n = len(items)
        pack_string = b'{}{}{}'.format(self.endianess, n, self.precision)
        return struct.pack(pack_string,*items)
    def __init__(self,filename):
        """set defaults"""
        pass
    def read_header(self):
        """read file header or fails"""
        return ('f',8,4,4,4) # open file
    def read_data(self,t,x,y,z):
        """random access read"""
        return (0,0,0,0,'data')
    def write_header(self,precision,nt,nx,ny,nz):
        """write header for new file"""
        pass
    def write_data(self,data):
        """write next site variables, in order"""
        pass
    def close(self):
        """closes the file"""
        self.file.close()

class GaugeCold(QCDFormat):
    def __init__(self,nt = 8,nx = 4,ny = 4,nz = 4):
        self.precision = 'f'
        self.size = (nt,nx,ny,nz)
    def read_header(self):
        (nt,nx,ny,nz) = self.size
        return (self.precision,nt,nx,ny,nz)
    def read_data(self,t,x,y,z):
        (nt,nx,ny,nz) = self.size
        return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0]


class GaugeMDP(QCDFormat):
    site_order = [T,X,Y,Z]
    link_order = [T,X,Y,Z]
    def __init__(self,filename,dummyfilename = 'none'):
        self.filename = filename
        self.dummyfilename = dummyfilename
        self.header_format = '<60s60s60sLi10iii'
        self.endianess = '<'
        self.header_size = 60+60+60+14*4
        self.offset = None
        self.site_size = None
        self.base_size = 4*9*2
    def read_header(self):
        self.file = open(self.filename,'rb')
        header = self.file.read(self.header_size)
        items = struct.unpack(b''.format(self.header_format),header)
        if items[3] != 1325884739:
            pass # should this raise exception?
        nt,nx,ny,nz = items[5:9]
        self.site_size = items[15]
        if self.site_size == self.base_size*4:
            self.precision = 'f'
        elif self.site_size == self.base_size*8:
            self.precision = 'd'
        else:
            raise IOError("unable to determine input precision")
        self.offset = self.file.tell()
        self.size = (nt,nx,ny,nz)
        return (self.precision,nt,nx,ny,nz)
    def write_header(self,precision,nt,nx,ny,nz):
        self.file = open(self.filename,'wb')
        self.site_size = self.base_size*(4 if precision == 'f' else 8)
        data = struct.pack(b''.format(self.header_format),'File Type: MDP FIELD',
                           self.dummyfilename,NOW.isoformat(),
                           1325884739,4,nt,nx,ny,nz,0,0,0,0,0,0,
                           self.site_size,nt*nx*ny*nz)
        self.file.write(data)
        self.size = (nt,nx,ny,nz)
        self.precision = precision
        self.offset = self.file.tell()
    def read_data(self,t,x,y,z):
        (nt,nx,ny,nz) = self.size
        i = self.offset + (z+nz*(y+ny*(x+nx*t)))*self.site_size
        self.file.seek(i)
        data = self.file.read(self.site_size)
        return self.unpack(data)
    def write_data(self,data,target_precision = None):
        if len(data) != self.base_size:
            raise RuntimeError("invalid data size")
        return self.file.write(self.pack(data))
    def convert_from(self,other,target_precision = None):
        (precision,nt,nx,ny,nz) = other.read_header()
        notify('  (precision: %s, size: %ix%ix%ix%i)' % (precision,nt,nx,ny,nz))
        self.write_header(target_precision or precision,nt,nx,ny,nz)
        pbar = ProgressBar(widgets = default_widgets , maxval = self.size[0]).start()
        for t in xrange(nt):
            for x in xrange(nx):
                for y in xrange(ny):
                    for z in xrange(nz):
                        data = other.read_data(t,x,y,z)
                        self.write_data(data)
            pbar.update(t)
        pbar.finish()


class GaugeMDPSplit(GaugeMDP):
    def convert_from(self,other,target_precision = None):
        (precision,nt,nx,ny,nz) = other.read_header()
        notify('  (precision: %s, size: %ix%ix%ix%i)' % (precision,nt,nx,ny,nz))
        pbar = ProgressBar(widgets = default_widgets , maxval = nt).start()
        for t in xrange(nt):
            slice = GaugeMDP(self.filename.replace('split.mdp',
                                                   't%.4i.mdp' % t))
            slice.write_header(target_precision or precision,1,nx,ny,nz)
            for x in xrange(nx):
                for y in xrange(ny):
                    for z in xrange(nz):
                        data = other.read_data(t,x,y,z)
                        slice.write_data(data)
            slice.close()
            pbar.update(t)
        pbar.finish()


class PropagatorMDP(QCDFormat):
    site_order = [T,X,Y,Z]
    is_gauge = False
    def __init__(self,filename):
        self.filename = filename
        self.header_format = '<60s60s60sLi10iii'
        self.endianess = '<'
        self.header_size = 60+60+60+14*4
        self.offset = None
        self.site_size = None
        self.base_size = 16*9*2
    def read_header(self):
        self.file = open(self.filename,'rb')
        header = self.file.read(self.header_size)
        items = struct.unpack(b''.format(self.header_format),header)
        if items[3] != '1325884739':
            pass # should this raise exception
        nt,nx,ny,nz = items[5:9]
        self.site_size = items[15]
        if self.site_size == self.base_size*4:
            self.precision = 'f'
        elif self.site_size == self.base_size*8:
            self.precision = 'd'
        else:
            raise IOError("file not in GaugeMDP format")
        self.offset = self.file.tell()
        self.size = (nt,nx,ny,nz)
        return (self.precision,nt,nx,ny,nz)
    def write_header(self,precision,nt,nx,ny,nz):
        self.file = open(self.filename,'wb')
        self.site_size = self.base_size*(4 if precision == 'f' else 8)
        data = struct.pack(b''.format(self.header_format),'File Type: MDP FIELD',
                           self.filename,NOW.isoformat(),
                           1325884739,4,nt,nx,ny,nz,0,0,0,0,0,0,
                           self.site_size,nt*nx*ny*nz)
        self.file.write(data)
        self.size = (nt,nx,ny,nz)
        self.precision = precision
        self.offset = self.file.tell()
    def read_data(self,t,x,y,z):
        i = self.offset + (z+nz*(y+ny*(x+nx*t)))*self.site_size
        self.file.seek(i)
        data = self.file.read(self.site_size)
        return self.unpack(data)
    def write_data(self,data,target_precision = None):
        if len(data) != self.base_size:
            raise RuntimeError("invalid data size")
        return self.file.write(self.pack(data))
    def convert_from(self,other,target_precision = None):
        (precision,nt,nx,ny,nz) = other.read_header()
        notify('  (precision: %s, size: %ix%ix%ix%i)' % (precision,nt,nx,ny,nz))
        self.write_header(target_precision or precision,nt,nx,ny,nz)
        pbar = ProgressBar(widgets = default_widgets , maxval = self.size[0]).start()
        for t in xrange(nt):
            for x in xrange(nx):
                for y in xrange(ny):
                    for z in xrange(nz):
                        data = other.read_data(t,x,y,z)
                        self.write_data(data)
            pbar.update(t)
        pbar.finish()


class PropagatorMDPSplit(QCDFormat):
    site_order = [T,X,Y,Z]
    is_gauge = False
    def __init__(self,filename):
        self.filename = filename
        self.header_format = '<60s60s60sLi10iii'
        self.endianess = '<'
        self.header_size = 60+60+60+14*4
        self.offset = None
        self.site_size = None
        self.base_size = 16*9*2
    def write_header(self,precision,nt,nx,ny,nz):
        self.file = open(self.filename,'wb')
        self.site_size = self.base_size*(4 if precision == 'f' else 8)
        data = struct.pack(b''.format(self.header_format),'File Type: MDP FIELD',
                           self.filename,NOW.isoformat(),
                           1325884739,4,nt,nx,ny,nz,0,0,0,0,0,0,
                           self.site_size,nt*nx*ny*nz)
        self.file.write(data)
        self.size = (nt,nx,ny,nz)
        self.precision = precision
        self.offset = self.file.tell()
    def write_data(self,data,target_precision = None):
        if len(data) != self.base_size:
            raise RuntimeError("invalid data size")
        return self.file.write(self.pack(data))
    def convert_from(self,other,target_precision = None):
        (precision,nt,nx,ny,nz) = other.read_header()
        notify('  (precision: %s, size: %ix%ix%ix%i)' % (precision,nt,nx,ny,nz))
        pbar = ProgressBar(widgets = default_widgets , maxval = nt).start()
        for t in xrange(nt):
            slice = PropagatorMDP(self.filename.replace('.split.prop.mdp',
                                                        '.t%.4i.prop.mdp' % t))
            slice.write_header(target_precision or precision,1,nx,ny,nz)
            for x in xrange(nx):
                for y in xrange(ny):
                    for z in xrange(nz):
                        data = other.read_data(t,x,y,z)
                        slice.write_data(data)
            slice.close()
            pbar.update(t)
        pbar.finish()


class GaugeILDG(QCDFormat):
    def __init__(self,filename,lfn = 'unkown'):
        self.filename = filename
        self.endianess = '>'
        self.lfn = lfn
        self.field = 'su3gauge'
        self.base_size = 4*9*2
    def read_header(self):
        self.lime = Lime(self.filename,'rb')
        self.file = self.lime.file
        for name,stream,size in self.lime:
            if name == 'ildg-binary-data':
                self.offset = stream.tell()
        for name,stream,size in self.lime:
            if name == 'ildg-format':
                data = stream.read(size)
                ### the following line is very important
                # The ILDG format computes the record size of non binary data
                # including the terminating zero of the C-style string representation
                # this is potentially a serious security vulnerability
                # of the ILDG file format.
                while data.endswith('\0'): data = data[:-1] # bug in generating data
                dxml = Lime.xml_parser(data)
                field = dxml("field")
                if field != self.field:
                    raise IOError('not a lime GaugeILDG')
                precision = int(dxml("precision"))
                nt = int(dxml("lt"))
                nx = int(dxml("lx"))
                ny = int(dxml("ly"))
                nz = int(dxml("lz"))
                if precision == 32:
                    self.precision = 'f'
                    self.site_size = self.base_size*4
                elif precision == 64:
                    self.precision = 'd'
                    self.site_size = self.base_size*8
                else:
                    raise IOError("unable to determine input precision")
                self.size = (nt,nx,ny,nz)
                return (self.precision,nt,nx,ny,nz)
        raise IOError("file is not in lime format")
    def write_header(self,precision,nt,nx,ny,nz):
        self.precision = precision
        self.site_size = 4*2*9*(4 if precision == 'f' else 8)
        self.size = (nt,nx,ny,nz)
        self.lime = Lime(self.filename,'wb')
        self.file = self.lime.file
        precision = 32 if precision == 'f' else 64
        d = dict(field = 'su3gauge',version = '1.0',
                 precision = precision,lx = nx,ly = ny,lz = nz,lt = nt)
        data = """<?xml version = "1.0" encoding = "UTF-8"?>
<ildgFormat>
<version>%(version)s</version>
<field>su3gauge</field>
<precision>%(precision)s</precision>
<lx>%(lx)s</lx><ly>%(ly)s</ly><lz>%(lz)s</lz><lt>%(lt)s</lt>
</ildgFormat>""" % d
        self.lime.write('ildg-format',data)
    def read_data(self,t,x,y,z):
        (nt,nx,ny,nz) = self.size
        i = self.offset + (x+nx*(y+ny*(z+nz*t)))*self.site_size
        self.file.seek(i)
        data = self.file.read(self.site_size)
        return self.unpack(data)
    def write_data(self,data,target_precision = None):
        if len(data) != self.base_size:
            raise RuntimeError("invalid data size")
        return self.file.write(self.pack(data))
    def convert_from(self,other,target_precision = None):
        (precision,nt,nx,ny,nz) = other.read_header()
        notify('  (precision: %s, size: %ix%ix%ix%i)' % (precision,nt,nx,ny,nz))
        self.write_header(target_precision or precision,nt,nx,ny,nz)
        pbar = ProgressBar(widgets = default_widgets , maxval = self.size[0]).start()
        def reader():
            for t in xrange(nt):
                for z in xrange(nz):
                    for y in xrange(ny):
                        for x in xrange(nx):
                            data = other.read_data(t,x,y,z)
                            yield self.pack(data)
                pbar.update(t)
        self.lime.write('ildg-binary-data',reader(),nt*nx*ny*nz*self.site_size)
        self.lime.write('ildg-data-LFN',self.lfn)
        self.lime.close()
        pbar.finish()


class GaugeSCIDAC(QCDFormat):
    def __init__(self,filename):
        self.filename = filename
        self.base_size = 4*9*2
        self.endianess = '>'
    def read_header(self):
        self.lime = Lime(self.filename,'rb')
        self.file = self.lime.file
        for name,stream,size in self.lime:
            if name == 'scidac-binary-data':
                self.offset = stream.tell()
        self.size = self.precision = None
        for name,stream,size in self.lime:
            if name == 'scidac-private-file-xml':
                data = stream.read(size)
                while data.endswith('\0'): data = data[:-1] # bug in generating data
                dxml = Lime.xml_parser(data)
                dims = dxml("dims").strip().split()
                nt = int(dims[3])
                nx = int(dims[0])
                ny = int(dims[1])
                nz = int(dims[2])
                self.size = (nt,nx,ny,nz)
        for name,stream,size in self.lime:
            if name == 'scidac-private-record-xml':
                data = stream.read(size)
                while data.endswith('\0'): data = data[:-1] # bug in generating data
                dxml = Lime.xml_parser(data)
                precision = dxml("precision").lower()
                if precision == 'f':
                    self.precision = 'f'
                    self.site_size = self.base_size*4
                elif precision == 'd':
                    self.precision = 'd'
                    self.site_size = self.base_size*8
                else:
                    raise IOError("unable to determine input precision")
        if self.size and self.precision:
            (nt,nx,ny,nz) = self.size
            return (self.precision,nt,nx,ny,nz)
        raise IOError("file is not in lime format")
    def read_data(self,t,x,y,z):
        (nt,nx,ny,nz) = self.size
        i = self.offset + (x+nx*(y+ny*(z+nz*t)))*self.site_size
        self.file.seek(i)
        data = self.file.read(self.site_size)
        return self.unpack(data)


class PropagatorSCIDAC(GaugeSCIDAC):
    is_gauge = False
    def __init__(self,filename):
        self.filename = filename
        self.base_size = 16*9*2
        self.endianess = '>'


class GaugeMILC(QCDFormat):
    def __init__(self,filename):
        self.filename = filename
        self.header_format = '<i4i64siii' # may change
        self.endianess = '<' # may change
        self.header_size = 96
        self.offset = None
        self.site_size = None
        self.base_size = 4*9*2
    def read_header(self):
        self.file = open(self.filename,'rb')
        header = self.file.read(self.header_size)
        for self.header_format in ('<i4i64siii','>i4i64siii'):
            self.endianess = self.header_format[0]
            items = struct.unpack(b''.format(self.header_format),header)
            if items[0] == 20103:
                nt,nx,ny,nz = [items[4],items[1],items[2],items[3]]
                self.site_size = (os.path.getsize(self.filename)-96) \
                  // nt // nx // ny // nz
                self.size = (nt,nx,ny,nz)
                if self.site_size == self.base_size*4:
                    self.precision = 'f'
                elif self.site_size == self.base_size*8:
                    self.precision = 'd'
                else:
                    raise IOError("file not in GaugeMILC fomat")
                self.offset = self.file.tell()
                self.size = (nt,nx,ny,nz)
                return (self.precision,nt,nx,ny,nz)
        raise IOError("file not in MILC format")
    def write_header(self,precision,nt,nx,ny,nz):
        self.file = open(self.filename,'wb')
        items = [9]
        items[0] = 20103
        items[1:5] = nx,ny,nz,nt
        items[5] = ""
        items[6] = 0
        items[7] = 0
        items[8] = 0
        header = struct.pack(b''.format(self.header_format),items)
        self.file.write(header)
        self.offset = self.file.tell()
    def read_data(self,t,x,y,z):
        (nt,nx,ny,nz) = self.size
        i = self.offset + (x+nx*(y+ny*(z+nz*t)))*self.site_size
        self.file.seek(i)
        data = self.file.read(self.site_size)
        return self.unpack(data)
    def write_data(self,data,target_precision = None):
        if len(data) != self.base_size:
            raise RuntimeError("invalid data size")
        return self.file.write(self.pack(data))
    def convert_from(self,other,target_precision = None):
        (precision,nt,nx,ny,nz) = other.read_header()
        notify('  (precision: %s, size: %ix%ix%ix%i)' % (precision,nt,nx,ny,nz))
        self.write_header(target_precision or precision,nt,nx,ny,nz)
        pbar = ProgressBar(widgets = default_widgets , maxval = self.size[0]).start()
        def reader():
            for t in xrange(nt):
                for z in xrange(nz):
                    for y in xrange(ny):
                        for x in xrange(nx):
                            data = other.read_data(t,x,y,z)
                            yield self.pack(data)
                pbar.update(t)

class GaugeNERSC(QCDFormat):
    def __init__(self,filename):
        self.filename = filename
        self.offset = None
        self.site_size = None
        self.base_size = 4*9*2
        self.endianess = '>'
    def read_header(self):
        self.file = open(self.filename,'rb')
        header = self.file.read(100000)
        self.offset = header.find('END_HEADER\n')+11
        if self.offset<20:
            raise IOError('not in nersc format')
        lines = header[:self.offset-1].split('\n')[1:-2]
        info = dict([[x.strip() for x in item.split(' = ',1)] for item in lines])
        nx = int(info['DIMENSION_1'])
        ny = int(info['DIMENSION_2'])
        nz = int(info['DIMENSION_3'])
        nt = int(info['DIMENSION_4'])
        if info['FLOATING_POINT'].endswith('SMALL'):
            self.endianess = '<'
        else: # assume default big endinan
            self.endianess = '>'
        if info['DATATYPE'] == '4D_SU3_GAUGE_3x3':
            self.reunitarize = False
        elif info['DATATYPE'] == '4D_SU3_GAUGE':
            self.reunitarize = True
            self.base_size = 4*6*2
        else:
            raise IOError("not in a known nersc format")
        if info['FLOATING_POINT'].startswith('IEEE32'):
            self.precision = 'f'
            self.site_size = self.base_size*4
        elif info['FLOATING_POINT'].startswith('IEEE64'):
            self.precision = 'd'
            self.site_size = self.base_size*8
        else:
            raise IOError("unable to determine input precision")
        self.size = (nt,nx,ny,nz)
        return (self.precision,nt,nx,ny,nz)
    def read_data(self,t,x,y,z):
        (nt,nx,ny,nz) = self.size
        i = self.offset + (x+nx*(y+ny*(z+nz*t)))*self.site_size
        self.file.seek(i)
        data = self.file.read(self.site_size)
        items = self.unpack(data)
        if self.reunitarize:
            new_items = []
            for i in range(4):
                new_items += reunitarize(items[i*12:(i+1)*12])
            items = new_items
        return items

OPTIONS = {
    'mdp':(GaugeMDP,GaugeMDP,GaugeMILC,GaugeNERSC,GaugeILDG,GaugeSCIDAC),
    'ildg':(GaugeILDG,GaugeILDG,GaugeMILC,GaugeNERSC,GaugeMDP,GaugeSCIDAC),
    'prop.mdp':(PropagatorMDP,PropagatorMDP,PropagatorSCIDAC),
    'prop.ildg':(PropagatorSCIDAC,PropagatorSCIDAC,PropagatorMDP),
    'split.mdp':(GaugeMDPSplit,GaugeMDP,GaugeMILC,GaugeNERSC,GaugeILDG,GaugeSCIDAC),
    'split.prop.mdp':(PropagatorMDPSplit,PropagatorMDP,PropagatorSCIDAC),
    }

ALL = (GaugeMDP,GaugeMILC,GaugeNERSC,GaugeILDG,GaugeSCIDAC,PropagatorMDP,PropagatorSCIDAC)

def universal_converter(path,target,precision,convert=True):
    filenames = [f for f in glob.glob(path) \
                     if not os.path.basename(f).startswith(CATALOG)]
    if not filenames:
        notify("no files to be converted")
        return
    processed = set()
    messages = []
    option = target and OPTIONS[target] or ALL
    for filename in filenames:
        for formatter in option[1:]:
            if convert:
                messages.append('trying to convert %s (%s)' %(filename,formatter.__name__))
            try:
                if convert:
                    ofilename = filename+'.'+target
                    if file_registered(ofilename):
                        notify('file %s already exists and is updated' % ofilename)
                    else:
                        dest = option[0](ofilename)
                        source = formatter(filename)
                        dest.convert_from(source,precision)
                        register_file(ofilename)
                else: # just pretend and get header info
                    info = formatter(filename).read_header()
                    notify('%s ... %s %s' % (filename,formatter.__name__,info))
                processed.add(filename)
                break
            except Exception(e):
                if convert:
                    messages.append('unable to convert:\n' + traceback.format_exc())
        if not filename in processed:
            if convert:
                notify('\n'.join(messages))
                sys.exit(1)
            else:
                notify('%s .... UNKOWN FORMAT' % filename)

