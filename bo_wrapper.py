from __future__ import print_function
import argparse
import os

import azure.storage.blob as azureblob
try:
   import cPickle as pickle
except:
   import pickle

from bo import bo

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--filepath', required=True,
                      help='The path to the text file to process. The path'
                           'may include a compute node\'s environment'
                           'variables, such as'
                           '$AZ_BATCH_NODE_SHARED_DIR/filename.txt')
  parser.add_argument('--storageaccount', required=True,
                      help='The name the Azure Storage account that owns the'
                           'blob storage container to which to upload'
                           'results.')
  parser.add_argument('--storagecontainer', required=True,
                      help='The Azure Blob storage container to which to'
                           'upload results.')
  parser.add_argument('--sastoken', required=True,
                      help='The SAS token providing write access to the'
                           'Storage container.')
  args = parser.parse_args()
  input_file = os.path.realpath(args.filepath)
  output_file = '{}_out{}'.format(
        os.path.splitext(args.filepath)[0],
        os.path.splitext(args.filepath)[1])
  parameter = pickle.load(open(input_file))
  #print(parameter)

  b = bo(*parameter)
  res = b.run()

  #print(res)
  pickle.dump(res, open(output_file, 'wb'))
  
  print("bo_wrapper.py listing files:")
  for item in os.listdir('.'):
    print(item)

  # Create the blob client using the container's SAS token.
  # This allows us to create a client that provides write
  # access only to the container.
  blob_client = azureblob.BlockBlobService(account_name=args.storageaccount, sas_token=args.sastoken)

  output_file_path = os.path.realpath(output_file)

  print('Uploading file {} to container [{}]...'.format(
        output_file_path,
        args.storagecontainer))

  blob_client.create_blob_from_path(args.storagecontainer,
                                      output_file,
                                      output_file_path)
