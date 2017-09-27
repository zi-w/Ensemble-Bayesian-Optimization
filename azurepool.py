from __future__ import print_function
try:
  import configparser
except ImportError:
  import ConfigParser as configparser
import datetime
import os

import azure.storage.blob as azureblob
import azure.batch.batch_service_client as batch
import azure.batch.batch_auth as batchauth
import azure.batch.models as batchmodels

import common.helpers
import time
try:
   import cPickle as pickle
except:
   import pickle
import sys
sys.path.append('.')

import logging

_TASK_FILE = 'bo_wrapper.py'

class AzurePool(object):
  def __init__(self, pool_id, data_dir):
    self.pool_id = pool_id
    if data_dir[-1] != '/':
      data_dir += '/'
    self.data_dir = data_dir
    self.app = common.helpers.generate_unique_resource_name('app')
    self.inp = common.helpers.generate_unique_resource_name('inp')
    self.out = common.helpers.generate_unique_resource_name('out')
    global_config = configparser.ConfigParser()
    global_config.read('configuration.cfg')

    our_config = configparser.ConfigParser()
    our_config.read('ebo.cfg')

    batch_account_key = global_config.get('Batch', 'batchaccountkey')
    batch_account_name = global_config.get('Batch', 'batchaccountname')
    batch_service_url = global_config.get('Batch', 'batchserviceurl')

    storage_account_key = global_config.get('Storage', 'storageaccountkey')
    storage_account_name = global_config.get('Storage', 'storageaccountname')
    storage_account_suffix = global_config.get(
      'Storage',
      'storageaccountsuffix')

    pool_vm_size = our_config.get(
      'DEFAULT',
      'poolvmsize')
    pool_vm_count = our_config.getint(
      'DEFAULT',
      'poolvmcount')
    
    # remember: no space, file names split by ','
    #app_file_names = our_config.get('APP', 'app').split(',')
    #app_file_names = [os.path.realpath(fn) for fn in app_file_names]
    # Print the settings we are running with
    common.helpers.print_configuration(global_config)
    common.helpers.print_configuration(our_config)

    credentials = batchauth.SharedKeyCredentials(
      batch_account_name,
      batch_account_key)
    batch_client = batch.BatchServiceClient(
      credentials,
      base_url=batch_service_url)

    # Retry 5 times -- default is 3
    batch_client.config.retry_policy.retries = 5
    
    self.storage_account_name = storage_account_name
    
    block_blob_client = azureblob.BlockBlobService(
      account_name=storage_account_name,
      account_key=storage_account_key,
      endpoint_suffix=storage_account_suffix)

    # create containers
    
    block_blob_client.create_container(self.app, fail_on_exist=False)
    block_blob_client.create_container(self.inp, fail_on_exist=False)
    block_blob_client.create_container(self.out, fail_on_exist=False)
    #app_files = upload_files(block_blob_client, self.app, app_file_names)
    
    output_container_sas_token = get_container_sas_token(
              block_blob_client,
              self.out,
              azureblob.BlobPermissions.WRITE)

    self.out_sas_token = output_container_sas_token

    create_pool(batch_client, pool_id, pool_vm_size, pool_vm_count, None)

    # create necessary folders
    if not os.path.exists(data_dir):
      os.makedirs(data_dir)

    self.batch_client = batch_client
    self.block_blob_client = block_blob_client
  def install(self):
    pool_id = self.pool_id
    batch_client = self.batch_client
    block_blob_client = self.block_blob_client
    job_id = common.helpers.generate_unique_resource_name(
          'install')
    run_commands(batch_client, block_blob_client, job_id, pool_id)
    common.helpers.wait_for_tasks_to_complete(
      batch_client,
      job_id,
      datetime.timedelta(minutes=25))

    tasks = batch_client.task.list(job_id)
    task_ids = [task.id for task in tasks]

    common.helpers.print_task_output(batch_client, job_id, task_ids)
  def end(self):
    self.delete_containers()
    self.delpool()
  def delpool(self):
    print("Deleting pool: ", self.pool_id)
    self.batch_client.pool.delete(self.pool_id)
  def delete_containers(self):
  	#pool_id, batch_client, block_blob_client
    # clean up
    self.block_blob_client.delete_container(
      self.inp,
      fail_not_exist=False)
    self.block_blob_client.delete_container(
      self.out,
      fail_not_exist=False)
    self.block_blob_client.delete_container(
      self.app,
      fail_not_exist=False)
  def reboot_failed_nodes(self):
    nodes = list(self.batch_client.compute_node.list(self.pool_id))
    failed = [n.id for n in nodes if n.state == batchmodels.ComputeNodeState.start_task_failed]

    for node in failed:
      self.batch_client.compute_node.reboot(self.pool_id, node)

  def reboot(self):
    nodes = list(self.batch_client.compute_node.list(self.pool_id))
    errored = [n.id for n in nodes if n.state == batchmodels.ComputeNodeState.unusable]
    working_nodes = [n.id for n in nodes if n not in errored]

    for node in working_nodes:
      self.batch_client.compute_node.reboot(self.pool_id, node)

  def map(self, parameters, job_id):
    # write parameters to files
    logging.info('In AzurePool map, job id [' + job_id + ']')
    batch_client = self.batch_client
    block_blob_client = self.block_blob_client
    job_id = common.helpers.generate_unique_resource_name(
          job_id)
    common.helpers.delete_blobs_from_container(block_blob_client, self.out)
    input_file_names = [os.path.join(self.data_dir,str(i) + '.pk') for i in xrange(len(parameters))]
    for i, p in enumerate(parameters):
      pickle.dump(p, open(input_file_names[i], 'wb'))
    #input_file_names = [os.path.realpath(fn) for fn in input_file_names]

    in_files = upload_files(block_blob_client, self.inp, input_file_names)

    # get app files again
    # remember: no blank line, one file each line
    app_file_names = get_list_from_file('py_files')
    #app_file_names = [os.path.realpath(fn) for fn in app_file_names]
    app_files = upload_files(block_blob_client, self.app, app_file_names)

    submit_job_and_add_tasks(batch_client, block_blob_client, job_id, self.pool_id, in_files, self.out, app_files, self.storage_account_name, self.out_sas_token)

    
    common.helpers.wait_for_tasks_to_complete(
      batch_client,
      job_id,
      datetime.timedelta(minutes=20))


    # GET outputs
    common.helpers.download_blobs_from_container(block_blob_client, self.out, './')

    #print(os.path.join(self.data_dir, str(0) + '_out.pk'))
    ret = []
    for i in xrange(len(parameters)):
      fnm = os.path.join(self.data_dir, str(i) + '_out.pk')
      if os.path.isfile(fnm):
        ret.append(pickle.load(open(fnm)))
      else:
        logging.warning('In AzurePool map, job id [' + job_id + '], ignoring lost parameter ' + str(i))
    if len(ret) == 0:
      try:
        tasks = batch_client.task.list(job_id)
        task_ids = [task.id for task in tasks]
        common.helpers.print_task_output(batch_client, job_id, task_ids)
        assert 0==1, 'No return from azure'
      except Exception as e:
        print('No return from azure and pring task output failed.')
        logging.error(e)
        raise e

    batch_client.job.delete(job_id)
    return ret


def upload_files(block_blob_client, container_name, files):
  #block_blob_client.create_container(container_name, fail_on_exist=False)
  return [get_resource_file(block_blob_client, container_name, \
    file_path, os.path.realpath(file_path)) for file_path in files]
  

def get_resource_file(block_blob_client, container_name, blob_name, file_path):
  sas_url = common.helpers.upload_blob_and_create_sas(
  block_blob_client,
  container_name,
  blob_name,
  file_path,
  datetime.datetime.utcnow() + datetime.timedelta(hours=1))
  logging.info('Uploading file {} from {} to container [{}]...'.format(blob_name, file_path, container_name))
  return batchmodels.ResourceFile(file_path=blob_name,
    blob_source=sas_url)
def get_list_from_file(file_nm):
  with open(file_nm) as f:
    content = f.readlines()
  return [x.strip() for x in content]

def create_pool(batch_client, pool_id, vm_size, vm_count, app_files):
  """Creates an Azure Batch pool with the specified id.

  :param batch_client: The batch client to use.
  :type batch_client: `batchserviceclient.BatchServiceClient`
  :param block_blob_client: The storage block blob client to use.
  :type block_blob_client: `azure.storage.blob.BlockBlobService`
  :param str pool_id: The id of the pool to create.
  :param str vm_size: vm size (sku)
  :param int vm_count: number of vms to allocate
  :param list app_files: The list of all the other scripts to upload.
  """
  # pick the latest supported 16.04 sku for UbuntuServer
  sku_to_use, image_ref_to_use = \
    common.helpers.select_latest_verified_vm_image_with_node_agent_sku(
      batch_client, 'Canonical', 'UbuntuServer', '14.04')
  user = batchmodels.AutoUserSpecification(
            scope=batchmodels.AutoUserScope.pool,
            elevation_level=batchmodels.ElevationLevel.admin)
  task_commands = get_list_from_file('start_commands')
  print(task_commands)
  pool = batchmodels.PoolAddParameter(
    id=pool_id,
    virtual_machine_configuration=batchmodels.VirtualMachineConfiguration(
      image_reference=image_ref_to_use,
      node_agent_sku_id=sku_to_use),
    vm_size=vm_size,
    target_dedicated=vm_count,
    start_task=batchmodels.StartTask(
      command_line=common.helpers.wrap_commands_in_shell('linux', task_commands),
      user_identity=batchmodels.UserIdentity(auto_user=user),
      resource_files=app_files,
      wait_for_success=True))

  common.helpers.create_pool_if_not_exist(batch_client, pool)

def run_commands(batch_client, block_blob_client, job_id, pool_id):
  """Run the start commands listed in the file "start_commands" on
  all the nodes of the Azure Batch service.

  :param batch_client: The batch client to use.
  :type batch_client: `batchserviceclient.BatchServiceClient`
  :param block_blob_client: The storage block blob client to use.
  :type block_blob_client: `azure.storage.blob.BlockBlobService`
  :param str job_id: The id of the job to create.
  :param str pool_id: The id of the pool to use.
  """
  task_commands = get_list_from_file('start_commands')
  logging.info(task_commands)
  user = batchmodels.AutoUserSpecification(
            scope=batchmodels.AutoUserScope.pool,
            elevation_level=batchmodels.ElevationLevel.admin)

  start = time.time()
  job = batchmodels.JobAddParameter(
    id=job_id,
    pool_info=batchmodels.PoolInformation(pool_id=pool_id))
  
  batch_client.job.add(job)
  logging.info('job created in seconds {}'.format(time.time() - start))

  start = time.time()
  nodes = list(batch_client.compute_node.list(pool_id))
  tasks = [batchmodels.TaskAddParameter(
    id="EBOTask-{}".format(i),
    command_line=common.helpers.wrap_commands_in_shell('linux', task_commands),
    user_identity=batchmodels.UserIdentity(auto_user=user)) \
    for i in xrange(len(nodes))]

  batch_client.task.add_collection(job.id, tasks)
  logging.info('task created in seconds {}'.format(time.time() - start))

def submit_job_and_add_tasks(batch_client, block_blob_client, job_id, pool_id, in_files, out_container_name, app_files, storage_account_name, out_sas_token):
  """Submits jobs to the Azure Batch service and adds
  tasks that runs a python script.

  :param batch_client: The batch client to use.
  :type batch_client: `batchserviceclient.BatchServiceClient`
  :param block_blob_client: The storage block blob client to use.
  :type block_blob_client: `azure.storage.blob.BlockBlobService`
  :param str job_id: The id of the job to create.
  :param str pool_id: The id of the pool to use.
  :param list in_files: The list of the file paths of the inputs.
  :param str out_container_name: The name of the output container.
  :param list app_files: The list of all the other scripts to upload.
  :param str storage_account_name: The name of the storage account.
  :param str out_sas_token: A SAS token granting the specified 
  permissions to the output container.
  """
  start = time.time()
  job = batchmodels.JobAddParameter(
    id=job_id,
    pool_info=batchmodels.PoolInformation(pool_id=pool_id))
  
  batch_client.job.add(job)
  logging.info('job created in seconds {}'.format(time.time() - start))

  start = time.time()
  
  tasks = [batchmodels.TaskAddParameter(
    id="EBOTask-{}".format(i),
    command_line='python {} --filepath {} --storageaccount {} --storagecontainer {} --sastoken "{}"'.format(_TASK_FILE,
             in_file.file_path,
             storage_account_name,
             out_container_name,
             out_sas_token),
    resource_files=[in_file] + app_files) \
    for i, in_file in enumerate(in_files)]

  cnt = 0
  tot_tasks = len(tasks)
  while cnt < tot_tasks:
    try:
      batch_client.task.add_collection(job.id, tasks[cnt:cnt+100])
      cnt += 100
    except Exception as e:
      print("Adding task failed... Going to try again in 5 seconds")
      logging.error(e)
      time.sleep(5)
  logging.info('task created in seconds {}'.format(time.time() - start))

def get_container_sas_token(block_blob_client,
  container_name, blob_permissions):
  """
  Obtains a shared access signature granting the specified permissions to the
  container.
  :param block_blob_client: A blob service client.
  :type block_blob_client: `azure.storage.blob.BlockBlobService`
  :param str container_name: The name of the Azure Blob storage container.
  :param BlobPermissions blob_permissions:
  :rtype: str
  :return: A SAS token granting the specified permissions to the container.
  """
  # Obtain the SAS token for the container, setting the expiry time and
  # permissions. In this case, no start time is specified, so the shared
  # access signature becomes valid immediately.
  container_sas_token = \
      block_blob_client.generate_container_shared_access_signature(
          container_name,
          permission=blob_permissions,
          expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=2))
  return container_sas_token
