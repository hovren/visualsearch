#!/usr/bin/env python3

import argparse
import sys
import os
import matplotlib.pyplot as plt
import time

# No toolbars
plt.rcParams['toolbar'] = 'None'

def matching_file(path):
    return os.path.splitext(path)[-1] == '.jpg'

class Labeler:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(24, 12))
        self.im = None
        self.fig.canvas.mpl_connect('key_press_event', self.key_cb)
        plt.show()
        
        self.waiting = True
        self.labels = []
    
    def toggle_label(self, l):
        if l in self.labels:
            self.labels.remove(l)
        else:
            self.labels.append(l)            
    
    def key_cb(self, event):
        if event.key == 'h':
            self.toggle_label('H')
        elif event.key == 'p':
            self.toggle_label('PE')
        elif event.key == " ": # Space
            self.waiting = False
        
    
    def label(self, path):
        self.path = path
        image = plt.imread(path)
        self.labels = []
        
        if self.im is None:
            self.im = self.ax.imshow(image)#, interpolation='none')
        else:
            self.im.set_data(image)
            
        self.fig.suptitle(path)
        self.fig.canvas.draw()
        
        self.waiting = True
        while self.waiting:
            self.ax.set_title('Labels: {}'.format(self.labels))
            plt.pause(0.1)
            #time.sleep(1.2)
            
        return self.labels                        
            
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    
    directory = os.path.expanduser(args.directory)
    labels_file = os.path.join(directory, 'labels.txt')
    
    if os.path.exists(labels_file) and not args.overwrite:
        print('{} already exists. Rerun with --overwrite.'.format(labels_file))
        sys.exit(-1)
        
    
    mfiles = [f for f in os.listdir(directory) if matching_file(f)]    

    labeler = Labeler()

    with open(labels_file, 'w') as lf:
        for f in sorted(mfiles):
            path = os.path.join(directory, f)
            labels = labeler.label(path)
            lf.write("{} {}\n".format(f, " ".join(labels)))
            
        
    
