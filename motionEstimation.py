#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

  This software is governed by the CeCILL-B license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL-B
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".

  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited
  liability.

  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.

  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL-B license and that you accept its terms.

"""
import argparse


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  parser.add_argument('-s', '--static', help='Static input image', type=str, required = True)
  parser.add_argument('-d', '--dyn', help='Dynamic 4D input image', type=str, required = True)
  parser.add_argument('-m', '--mask', help='Segmentation high-resolution mask image', type=str, required = True,action='append')
  parser.add_argument('-o', '--output', help='Output directory', type=str, required = True)

  args = parser.parse_args()

  
  
