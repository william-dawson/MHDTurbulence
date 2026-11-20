module mht_nccl
  use cudafor, only: cuda_stream_kind, cudaStreamSynchronize
  use mpi, only: MPI_Abort, MPI_Bcast, MPI_CHARACTER, MPI_SUCCESS
  use nccl
  use openacc, only: acc_async_sync, acc_get_cuda_stream
  implicit none
  private
  public :: mht_nccl_start, mht_nccl_finish, mht_nccl_reduce

  type(ncclComm), save :: device_comm
  logical, save :: wait_for_reduction = .true.

contains

  subroutine mht_nccl_start(comm, ranks, rank)
    integer, intent(in) :: comm, ranks, rank
    integer, parameter :: id_bytes = 128
    type(ncclUniqueId) :: id
    integer :: ierr, abort_status
    real(8) :: warm_input(2), warm_output(2)
    character(len=1) :: asynchronous
    if (rank == 0) call check(ncclGetUniqueId(id), comm, "ncclGetUniqueId")
    call MPI_Bcast(id%internal, id_bytes, MPI_CHARACTER, 0, comm, ierr)
    if (ierr /= MPI_SUCCESS) call MPI_Abort(comm, 1, abort_status)
    call check(ncclCommInitRank(device_comm, ranks, id, rank), comm, "ncclCommInitRank")
    warm_input = 0.0d0
!$acc data copyin(warm_input) create(warm_output)
    call mht_nccl_reduce(warm_input, warm_output, .true., comm)
    call mht_nccl_reduce(warm_input, warm_output, .false., comm)
!$acc end data
    call get_environment_variable("MHT_NCCL_ASYNC", asynchronous)
    wait_for_reduction = asynchronous /= '1'
  end subroutine mht_nccl_start

  subroutine mht_nccl_finish(comm)
    integer, intent(in) :: comm
    call check(ncclCommDestroy(device_comm), comm, "ncclCommDestroy")
  end subroutine mht_nccl_finish

  subroutine mht_nccl_reduce(input, output, minimum, comm)
    real(8), intent(in) :: input(2)
    real(8), intent(out) :: output(2)
    logical, intent(in) :: minimum
    integer, intent(in) :: comm
    integer(cuda_stream_kind) :: stream
    type(ncclResult) :: result
    integer :: ierr, abort_status

    stream = acc_get_cuda_stream(acc_async_sync)
!$acc host_data use_device(input, output)
    if (minimum) then
       result = ncclAllReduce(input, output, 1, ncclDouble, ncclMin, device_comm, stream)
    else
       result = ncclAllReduce(input, output, 1, ncclDouble, ncclMax, device_comm, stream)
    endif
!$acc end host_data
    call check(result, comm, "ncclAllReduce")
    ! The following OpenACC region consumes the result on the same stream.
    if (.not. wait_for_reduction) return
    ierr = cudaStreamSynchronize(stream)
    if (ierr /= 0) then
       print *, "cudaStreamSynchronize failed after NCCL reduction", ierr
       call MPI_Abort(comm, 1, abort_status)
    endif
  end subroutine mht_nccl_reduce

  subroutine check(result, comm, operation)
    type(ncclResult), intent(in) :: result
    integer, intent(in) :: comm
    character(*), intent(in) :: operation
    integer :: ierr
    if (result /= ncclSuccess) then
       print *, operation, " failed"
       call MPI_Abort(comm, 1, ierr)
    endif
  end subroutine check

end module mht_nccl
