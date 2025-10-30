function(BuildRule EXECUTABLE)
  # set sources to all parameters
  set(SOURCES ${ARGV})
  # remove the first argument (the executable)
  list(REMOVE_AT SOURCES 0)
  add_executable(${EXECUTABLE} ${SOURCES})
  target_link_libraries(
    ${EXECUTABLE}
    PRIVATE cudf::cudf
            ucxx
            ucp
            ucs
            fmt
            gflags)
endfunction()

buildrule(
  communicator
  commsMain.cpp
  Communicator.cpp
  Acceptor.cpp
  EndpointRef.cpp
  Sender.cpp
  Receiver.cpp
  protocol.cpp
  util.cpp)
