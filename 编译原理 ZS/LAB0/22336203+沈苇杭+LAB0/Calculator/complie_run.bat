@echo off

set SOURCE_DIR=src\main\java\
set BIN_DIR=bin

if not exist %BIN_DIR% mkdir %BIN_DIR%

echo Source directory: %SOURCE_DIR%
echo Output directory: %BIN_DIR%

echo Compiling TaxRange.java from: %SOURCE_DIR%TaxRange.java
echo Compiling TaxCmpt.java from: %SOURCE_DIR%TaxCmpt.java
echo Compiling Main.java from: %SOURCE_DIR%Main.java

javac -d %BIN_DIR% %SOURCE_DIR%TaxRange.java %SOURCE_DIR%TaxCmpt.java %SOURCE_DIR%Main.java

java -cp %BIN_DIR% Main

pause
